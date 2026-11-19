"""Stage 2: score generated robot videos with the Qwen3.8-27B vision-language judge.

For every video in a rollout manifest the judge sees K evenly sampled frames plus the clip's caption and
returns a 1-5 score with a one-sentence reason on four axes:

  task_completion       the instruction in the caption is carried out
  object_interaction    the gripper contacts and moves the right object
  physical_consistency  no vanishing / teleporting / interpenetrating / melting objects
  temporal_coherence    arm, objects and background keep a stable appearance over time

Output is one JSON line per video (resumable; videos already scored are skipped):
  {"clip", "seed", "video", "scores": {axis: int}, "reasons": {axis: str}, "total", "mean",
   "raw": model text, "parse_ok": bool, "latency_s", "judge": {...settings}}

Memory: the model is 56 GB in bf16. On a single 24 GB GPU use --quant nf4 (default), ~16 GB.
With two or more GPUs, --quant none --device-map auto runs it in bf16.

Examples:
  # score everything in the train rollout manifest
  CUDA_VISIBLE_DEVICES=1 python scripts/judge_videos.py --manifest outputs/rollouts/train/manifest.jsonl
  # score a handful of arbitrary files with a caption (reliability checks, val set)
  python scripts/judge_videos.py --videos a.mp4 b.mp4 --caption "The arm pushes the beans into a pile" --out s.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen3.8-27B"

AXES = ["task_completion", "object_interaction", "physical_consistency", "temporal_coherence"]

JUDGE_INSTRUCTIONS = """You are a strict evaluator of AI-generated robot manipulation videos.
You are shown {k} frames sampled evenly in time from a {dur:.1f}-second video of a WidowX robot arm on a tabletop.
Frame 1 is real; every later frame was produced by a video-generation model that was asked to continue the scene
according to this description:

DESCRIPTION:
{caption}

RULES
- Judge only what is visible in the frames. Never give credit for things that are implied but not shown.
- Generated videos often look fine at a glance. Before scoring, check frame by frame: count the movable objects
  in frame 1 and verify they are all still present, same size and same shape, in the last frame; follow the
  gripper and ask whether it ever actually touches the object it is supposed to manipulate; look for any object
  that appears, disappears, splits, merges, floats, or passes through the gripper or table.
- When torn between two scores, give the lower one.
- A score of 5 on any axis requires that you found no flaw at all on that axis.

SCALES

task_completion  (was the described task done?)
  5 = the described task is clearly completed by the final frames, with the right object ending in the right
      place or state, and the frames show the whole process, not just the end result.
  4 = the task is completed but with a minor deviation (slightly wrong placement, target not fully reached,
      one step rushed or ambiguous).
  3 = the arm clearly attempts the described task and makes partial progress (e.g. grasps or moves the object
      part of the way) but the task is not finished by the last frame.
  2 = the arm moves toward or near the relevant object, but no real progress on the task is visible.
  1 = nothing resembling the described task happens: the arm idles, wanders, does a different task, or the
      scene changes without the arm causing it.

object_interaction  (does the gripper physically interact with the correct object?)
  5 = the gripper makes visible contact with the correct object and the object moves in a way that matches the
      gripper motion (grasped objects travel with the gripper, pushed objects move in the push direction).
  4 = contact and consistent motion, but with one small implausibility (slight slip, slightly late or early
      motion, grasp looks loose).
  3 = the gripper touches the correct object, but the object's motion does not match the gripper (moves before
      contact, lags behind, moves too far, or stays still while the gripper moves as if holding it).
  2 = the gripper interacts with the wrong object, or only brushes the correct object with no effect.
  1 = no contact with any relevant object; or objects move with no gripper contact at all.

physical_consistency  (do objects persist and obey physics?)
  5 = every movable object in frame 1 is present in every later frame with the same size and shape; nothing
      appears, duplicates, floats, teleports, or interpenetrates; the robot arm keeps one consistent body.
  4 = one small violation only (a small object briefly flickers or is slightly deformed in one frame, a minor
      shadow or texture glitch).
  3 = one clear violation (one object disappears, appears, or visibly changes size/shape; one interpenetration
      of gripper and object or table).
  2 = two or three clear violations, or one severe one (the main target object vanishes, or an object appears
      from nowhere and is then manipulated).
  1 = objects vanish and reappear repeatedly, multiply, melt, or the arm splits, bends impossibly, or passes
      through the table.

temporal_coherence  (is the video visually stable from frame to frame?)
  5 = arm, objects and background keep a consistent appearance throughout; motion between consecutive frames is
      smooth and small; the camera does not move.
  4 = minor texture flicker or blur on fast-moving parts, nothing else.
  3 = one abrupt jump (an object or the arm is in a clearly different place in one frame than the previous and
      next frames would allow), or noticeable morphing of textures or edges.
  2 = several jumps or persistent morphing; the background or table changes appearance.
  1 = the frames do not read as one continuous video (scene cuts, large position jumps, objects changing identity).

PROCEDURE  (do this before scoring; your scores must agree with these observations)
1. In frame 1, list every movable object and, for small repeated items (beads, beans, gems, blocks), estimate
   their count. Do the same for a MIDDLE frame (frame {mid}) and for the LAST frame. Compare all three. If a
   count drops or rises by more than about 20% at any of these points, or an object present in frame 1 is missing
   in any later frame, or a new object is present that was not in frame 1, that is an object persistence
   violation and physical_consistency cannot be higher than 3 (2 if the missing or new object is the one being
   manipulated, or if many items vanished). Objects that vanish and later reappear are a violation even though
   the first and last frames match.
2. For each frame, note where the gripper is and whether its fingers are touching an object. "Touching" means the
   object is between or against the fingers, not merely below the gripper.
3. Note whether the object the gripper touches moves together with the gripper in the following frames.
4. Only then assign the four scores using the scales above.

Respond with ONLY a JSON object, no prose before or after, in exactly this form:
{{"observations": {{
    "frame_1_objects": "<objects with approximate counts>",
    "middle_frame_objects": "<objects with approximate counts>",
    "last_frame_objects": "<objects with approximate counts>",
    "gripper_contact": "<which frames the fingers touch which object, or 'none'>",
    "anomalies": "<objects that vanish, appear, duplicate, morph, teleport, or 'none'>"}},
 "task_completion": {{"score": <1-5>, "reason": "<one sentence naming the specific evidence>"}},
 "object_interaction": {{"score": <1-5>, "reason": "<one sentence naming the specific evidence>"}},
 "physical_consistency": {{"score": <1-5>, "reason": "<one sentence naming the specific evidence>"}},
 "temporal_coherence": {{"score": <1-5>, "reason": "<one sentence naming the specific evidence>"}}}}"""


# --------------------------------------------------------------------------- video

def sample_frames(video_path: str, k: int) -> tuple[list[Image.Image], float, float]:
    """Return k evenly spaced RGB frames, the source fps and the clip duration in seconds."""
    import av

    frames: list[Image.Image] = []
    with av.open(video_path) as c:
        stream = c.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 5.0
        all_frames = [f.to_image().convert("RGB") for f in c.decode(video=0)]
    n = len(all_frames)
    if n == 0:
        raise RuntimeError(f"no frames in {video_path}")
    idx = np.linspace(0, n - 1, min(k, n)).round().astype(int)
    frames = [all_frames[i] for i in idx]
    return frames, fps, n / fps


# --------------------------------------------------------------------------- model

def load_judge(model_dir: Path, quant: str, device_map: str, attn: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs: dict = {"dtype": torch.bfloat16, "device_map": device_map}
    if attn:
        kwargs["attn_implementation"] = attn
    if quant == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "int8":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant != "none":
        raise ValueError(f"unknown --quant {quant}")

    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **kwargs)
    model.eval()
    return processor, model


def build_inputs(processor, frames: list[Image.Image], sample_fps: float, prompt: str, thinking: bool,
                 as_images: bool = False, duration: float | None = None):
    """Tokenize one request.

    as_images=False: frames go in as one video (Qwen merges 2x2 patches and frame pairs: cheap, but small
                     objects blur away at 256 px).
    as_images=True:  each frame is a separately labelled image ("Frame k of N, t=..s"), which keeps per-frame
                     detail and lets the judge refer to frames by number.
    """
    common = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                  enable_thinking=thinking)
    if as_images:
        n = len(frames)
        dur = duration if duration else n / max(sample_fps, 1e-6)
        content: list[dict] = []
        for i, fr in enumerate(frames):
            t = (i / max(n - 1, 1)) * dur
            content.append({"type": "text", "text": f"Frame {i + 1} of {n} (t = {t:.1f} s):"})
            content.append({"type": "image", "image": fr})
        content.append({"type": "text", "text": prompt})
        return processor.apply_chat_template([{"role": "user", "content": content}], **common)

    # Video path. The frames are ALREADY sampled, so frame resampling must be disabled and real timing
    # metadata supplied; otherwise Qwen3-VL's video processor assumes 24 fps, treats 16 frames as a 0.7 s
    # clip and keeps only two of them (verified: grid_thw [2,16,16] instead of [8,32,32]).
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=thinking)
    video = np.stack([np.asarray(f) for f in frames])  # [T,H,W,3] uint8
    n = len(frames)
    dur = duration if duration else n / max(sample_fps, 1e-6)
    meta = {"fps": n / dur, "total_num_frames": n, "duration": dur, "frames_indices": list(range(n))}
    inputs = processor(
        text=[text],
        videos=[video],
        videos_kwargs={"video_metadata": [meta], "do_sample_frames": False},
        return_tensors="pt",
    )
    thw = inputs.get("video_grid_thw")
    if thw is not None and int(thw[0][0]) * 2 < n:  # temporal patch size is 2
        raise RuntimeError(f"video processor kept {int(thw[0][0]) * 2} of {n} frames; metadata not honoured")
    return inputs


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_scores(text: str) -> tuple[dict[str, int], dict[str, str], bool]:
    """Extract the four scores; tolerate <think> blocks, code fences and trailing prose."""
    body = text.split("</think>")[-1] if "</think>" in text else text
    body = body.replace("```json", "").replace("```", "")
    m = _JSON_RE.search(body)
    scores: dict[str, int] = {}
    reasons: dict[str, str] = {}
    ok = False
    if m:
        try:
            obj = json.loads(m.group(0))
            for ax in AXES:
                v = obj.get(ax)
                if isinstance(v, dict):
                    scores[ax] = int(round(float(v.get("score"))))
                    reasons[ax] = str(v.get("reason", "")).strip()
                elif v is not None:
                    scores[ax] = int(round(float(v)))
                    reasons[ax] = ""
            ok = all(ax in scores and 1 <= scores[ax] <= 5 for ax in AXES)
        except Exception:
            ok = False
    if not ok:
        # last resort: "task_completion": {"score": 4  /  task_completion: 4
        for ax in AXES:
            mm = re.search(rf"{ax}\"?\s*:\s*(?:\{{\s*\"score\"\s*:\s*)?([1-5])", body)
            if mm:
                scores[ax] = int(mm.group(1))
                reasons.setdefault(ax, "")
        ok = all(ax in scores for ax in AXES)
    return scores, reasons, ok


@torch.inference_mode()
def judge_one(processor, model, video_path: str, caption: str, *, k: int, thinking: bool,
              max_new_tokens: int, temperature: float, frame_size: int = 0, as_images: bool = False) -> dict:
    frames, src_fps, duration = sample_frames(video_path, k)
    if frame_size and frames[0].size[0] != frame_size:
        # Upscale so Qwen's 32-px-per-token grid still resolves beads, gems and gripper fingers.
        frames = [f.resize((frame_size, frame_size), Image.BICUBIC) for f in frames]
    sample_fps = len(frames) / duration if duration > 0 else 1.0
    prompt = JUDGE_INSTRUCTIONS.format(k=len(frames), dur=duration, caption=caption.strip(),
                                       mid=len(frames) // 2 + 1)
    inputs = build_inputs(processor, frames, sample_fps, prompt, thinking, as_images=as_images, duration=duration)
    inputs = inputs.to(model.device)

    gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.8, top_k=20)
    else:
        gen_kwargs.update(do_sample=False)
    t0 = time.time()
    out = model.generate(**inputs, **gen_kwargs)
    latency = time.time() - t0
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    scores, reasons, ok = parse_scores(text)
    observations = None
    m = _JSON_RE.search(text.split("</think>")[-1].replace("```json", "").replace("```", ""))
    if m:
        try:
            observations = json.loads(m.group(0)).get("observations")
        except Exception:
            observations = None
    rec = {
        "scores": scores,
        "reasons": reasons,
        "observations": observations,
        "total": sum(scores.values()) if ok else None,
        "mean": round(sum(scores.values()) / len(scores), 3) if ok else None,
        "parse_ok": ok,
        "raw": text,
        "latency_s": round(latency, 2),
        "n_frames": len(frames),
        "src_fps": src_fps,
        "duration_s": round(duration, 2),
        "n_new_tokens": int(new_tokens.numel()),
    }
    return rec



def _prepare_one(processor, video_path: str, caption: str, *, k: int, thinking: bool, frame_size: int,
                 as_images: bool):
    frames, src_fps, duration = sample_frames(video_path, k)
    if frame_size and frames[0].size[0] != frame_size:
        frames = [f.resize((frame_size, frame_size), Image.BICUBIC) for f in frames]
    sample_fps = len(frames) / duration if duration > 0 else 1.0
    prompt = JUDGE_INSTRUCTIONS.format(k=len(frames), dur=duration, caption=caption.strip(),
                                       mid=len(frames) // 2 + 1)
    inputs = build_inputs(processor, frames, sample_fps, prompt, thinking, as_images=as_images, duration=duration)
    meta = {"n_frames": len(frames), "src_fps": src_fps, "duration_s": round(duration, 2)}
    return inputs, meta


def _finish_record(processor, new_tokens: torch.Tensor, meta: dict, latency: float) -> dict:
    text = processor.decode(new_tokens, skip_special_tokens=True)
    scores, reasons, ok = parse_scores(text)
    observations = None
    m = _JSON_RE.search(text.split("</think>")[-1].replace("```json", "").replace("```", ""))
    if m:
        try:
            observations = json.loads(m.group(0)).get("observations")
        except Exception:
            observations = None
    return {
        "scores": scores,
        "reasons": reasons,
        "observations": observations,
        "total": sum(scores.values()) if ok else None,
        "mean": round(sum(scores.values()) / len(scores), 3) if ok else None,
        "parse_ok": ok,
        "raw": text,
        "latency_s": round(latency, 2),
        "n_new_tokens": int(new_tokens.numel()),
        **meta,
    }


@torch.inference_mode()
def judge_batch(processor, model, jobs: list[dict], *, k: int, thinking: bool, max_new_tokens: int,
                temperature: float, frame_size: int = 512, as_images: bool = True) -> list[dict]:
    """Score several videos in one generate() call. Sequences are left-padded so generation aligns."""
    prepared = [_prepare_one(processor, j["video"], j["caption"], k=k, thinking=thinking,
                             frame_size=frame_size, as_images=as_images) for j in jobs]
    if len(prepared) == 1:
        inputs, meta = prepared[0]
        inputs = inputs.to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        t0 = time.time()
        out = model.generate(**inputs, **_gen_kwargs(max_new_tokens, temperature))
        return [_finish_record(processor, out[0, prompt_len:], meta, time.time() - t0)]

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    ids_list = [p[0]["input_ids"][0] for p in prepared]
    max_len = max(x.shape[0] for x in ids_list)
    input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=ids_list[0].dtype)
    attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
    for i, x in enumerate(ids_list):  # left pad
        input_ids[i, max_len - x.shape[0]:] = x
        attention_mask[i, max_len - x.shape[0]:] = 1
    batch_inputs: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
    # Vision tensors are concatenated along the patch axis; grid_thw rows along the batch axis.
    for key in ("pixel_values", "pixel_values_videos"):
        if key in prepared[0][0]:
            batch_inputs[key] = torch.cat([p[0][key] for p in prepared], dim=0)
    for key in ("image_grid_thw", "video_grid_thw"):
        if key in prepared[0][0]:
            batch_inputs[key] = torch.cat([p[0][key] for p in prepared], dim=0)
    batch_inputs = {k_: v.to(model.device) for k_, v in batch_inputs.items()}
    t0 = time.time()
    out = model.generate(**batch_inputs, **_gen_kwargs(max_new_tokens, temperature))
    latency = (time.time() - t0) / len(prepared)
    recs = []
    for i, (_, meta) in enumerate(prepared):
        new_tokens = out[i, max_len:]
        # strip trailing pads produced after EOS in the batch
        eos = processor.tokenizer.eos_token_id
        if pad_id is not None:
            keep = (new_tokens != pad_id)
            if eos is not None:
                keep &= (new_tokens != eos) | (torch.cumsum((new_tokens == eos).long(), 0) <= 1)
            new_tokens = new_tokens[keep]
        recs.append(_finish_record(processor, new_tokens, meta, latency))
    return recs


def _gen_kwargs(max_new_tokens: int, temperature: float) -> dict:
    gk: dict = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gk.update(do_sample=True, temperature=temperature, top_p=0.8, top_k=20)
    else:
        gk.update(do_sample=False)
    return gk

# --------------------------------------------------------------------------- main

def load_jobs(args) -> list[dict]:
    jobs: list[dict] = []
    if args.manifest:
        caption_cache: dict[str, str] = {}
        with open(args.manifest) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                pf = r.get("prompt_file")
                if pf and pf not in caption_cache:
                    p = json.loads(Path(pf).read_text())
                    caption_cache[pf] = p.get("caption") or p.get("prompt")
                jobs.append({"clip": r["clip"], "seed": r.get("seed"), "video": r["video"],
                             "caption": caption_cache.get(pf, r.get("caption", ""))})
    if args.videos:
        if not args.caption:
            raise SystemExit("--videos requires --caption")
        for v in args.videos:
            jobs.append({"clip": Path(v).stem, "seed": None, "video": v, "caption": args.caption})
    if args.clips:
        wanted = set(args.clips.split(","))
        jobs = [j for j in jobs if j["clip"] in wanted]
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    jobs = [j for i, j in enumerate(jobs) if i % shard_n == shard_i]
    if args.limit:
        jobs = jobs[: args.limit]
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("inputs")
    src.add_argument("--manifest", type=Path, help="rollout manifest.jsonl from generate_rollouts.py")
    src.add_argument("--videos", nargs="*", help="explicit video files (with --caption)")
    src.add_argument("--caption", help="caption used for all --videos")
    src.add_argument("--clips", help="comma-separated clip ids to restrict to")
    src.add_argument("--limit", type=int)
    src.add_argument("--shard", default="0/1")
    ap.add_argument("--out", type=Path, help="scores JSONL (default: <manifest dir>/judge_scores.jsonl)")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--quant", choices=["nf4", "int8", "none"], default="nf4")
    ap.add_argument("--device-map", default="cuda:0", help="'cuda:0' for one GPU, 'auto' to spread over several")
    ap.add_argument("--attn", default="sdpa", help="attention implementation ('' = library default)")
    ap.add_argument("--frames", type=int, default=16, help="frames sampled per video")
    ap.add_argument("--frame-size", type=int, default=512,
                    help="upscale frames to this square size before the judge sees them (0 = native)")
    ap.add_argument("--as-video", action="store_true",
                    help="send frames as one video (cheaper, but Qwen merges frame pairs and 2x2 patches, which "
                         "hides small objects). Default is one labelled image per frame.")
    ap.add_argument("--batch-size", type=int, default=1, help="videos scored per forward pass (left-padded batch)")
    ap.add_argument("--thinking", action="store_true", help="enable Qwen thinking mode (slower, more tokens)")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (reproducible scores)")
    args = ap.parse_args()

    if not args.manifest and not args.videos:
        raise SystemExit("give --manifest or --videos")
    out = args.out or (args.manifest.parent / "judge_scores.jsonl" if args.manifest else Path("judge_scores.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    max_new = args.max_new_tokens or (2048 if args.thinking else 700)

    jobs = load_jobs(args)
    done = set()
    if out.exists():
        with open(out) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("parse_ok"):
                        done.add(r["video"])
    todo = [j for j in jobs if j["video"] not in done]
    print(f"videos={len(jobs)} already_scored={len(done)} to_score={len(todo)} -> {out}")
    if not todo:
        return

    t0 = time.time()
    processor, model = load_judge(args.model_dir, args.quant, args.device_map, args.attn)
    print(f"judge loaded in {time.time() - t0:.0f}s; gpu mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    settings = {"model": args.model_dir.name, "quant": args.quant, "frames": args.frames,
                "frame_size": args.frame_size, "as_images": not args.as_video, "batch_size": args.batch_size,
                "thinking": args.thinking, "temperature": args.temperature, "max_new_tokens": max_new}
    n = 0
    n_bad = 0
    t_start = time.time()
    bs = max(1, args.batch_size)
    with open(out, "a") as f:
        for start in range(0, len(todo), bs):
            batch = todo[start:start + bs]
            try:
                recs = judge_batch(processor, model, batch, k=args.frames, thinking=args.thinking,
                                   max_new_tokens=max_new, temperature=args.temperature,
                                   frame_size=args.frame_size, as_images=not args.as_video)
            except Exception as e:  # keep going; record the failure for every video in the batch
                recs = [{"scores": {}, "reasons": {}, "observations": None, "total": None, "mean": None,
                         "parse_ok": False, "raw": f"ERROR: {type(e).__name__}: {e}", "latency_s": None}
                        for _ in batch]
            for j, rec in zip(batch, recs):
                rec.update({"clip": j["clip"], "seed": j["seed"], "video": j["video"], "judge": settings})
                f.write(json.dumps(rec) + "\n")
                n += 1
                n_bad += 0 if rec["parse_ok"] else 1
            f.flush()
            if (n // bs) % max(1, 10 // bs) == 0 or n >= len(todo):
                avg = (time.time() - t_start) / n
                print(f"[{n}/{len(todo)}] {batch[-1]['clip']} seed={batch[-1]['seed']} total={recs[-1]['total']} "
                      f"avg={avg:.1f}s/video eta={avg * (len(todo) - n) / 3600:.1f}h parse_fail={n_bad} "
                      f"peak_mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
    print(f"done: {n} scored, {n_bad} parse failures -> {out}")


if __name__ == "__main__":
    main()
