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

Respond with ONLY a JSON object, no prose before or after, in exactly this form:
{{"task_completion": {{"score": <1-5>, "reason": "<one sentence naming the specific evidence>"}},
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


def build_inputs(processor, frames: list[Image.Image], sample_fps: float, prompt: str, thinking: bool):
    """Tokenize one (video, prompt) request. Frames go in as a video so Qwen applies its temporal patching."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    common = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                  enable_thinking=thinking)
    try:
        # Preferred path: let the chat template handle frame lists + metadata.
        return processor.apply_chat_template(messages, fps=sample_fps, **common)
    except Exception:
        # Fallback: render text ourselves and hand the processor a [T,H,W,C] uint8 array.
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                             enable_thinking=thinking)
        video = np.stack([np.asarray(f) for f in frames])  # [T,H,W,3]
        meta = {"fps": sample_fps, "total_num_frames": len(frames), "duration": len(frames) / sample_fps,
                "frames_indices": list(range(len(frames)))}
        try:
            return processor(text=[text], videos=[video], video_metadata=[meta], return_tensors="pt")
        except TypeError:
            return processor(text=[text], videos=[video], return_tensors="pt")


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
              max_new_tokens: int, temperature: float) -> dict:
    frames, src_fps, duration = sample_frames(video_path, k)
    sample_fps = len(frames) / duration if duration > 0 else 1.0
    prompt = JUDGE_INSTRUCTIONS.format(k=len(frames), dur=duration, caption=caption.strip())
    inputs = build_inputs(processor, frames, sample_fps, prompt, thinking)
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
    rec = {
        "scores": scores,
        "reasons": reasons,
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
    ap.add_argument("--thinking", action="store_true", help="enable Qwen thinking mode (slower, more tokens)")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (reproducible scores)")
    args = ap.parse_args()

    if not args.manifest and not args.videos:
        raise SystemExit("give --manifest or --videos")
    out = args.out or (args.manifest.parent / "judge_scores.jsonl" if args.manifest else Path("judge_scores.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    max_new = args.max_new_tokens or (2048 if args.thinking else 400)

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
                "thinking": args.thinking, "temperature": args.temperature, "max_new_tokens": max_new}
    n = 0
    n_bad = 0
    t_start = time.time()
    with open(out, "a") as f:
        for j in todo:
            try:
                rec = judge_one(processor, model, j["video"], j["caption"], k=args.frames, thinking=args.thinking,
                                max_new_tokens=max_new, temperature=args.temperature)
            except Exception as e:  # keep going; record the failure
                rec = {"scores": {}, "reasons": {}, "total": None, "mean": None, "parse_ok": False,
                       "raw": f"ERROR: {type(e).__name__}: {e}", "latency_s": None}
            rec.update({"clip": j["clip"], "seed": j["seed"], "video": j["video"], "judge": settings})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            n_bad += 0 if rec["parse_ok"] else 1
            if n % 10 == 0 or n == len(todo):
                avg = (time.time() - t_start) / n
                print(f"[{n}/{len(todo)}] {j['clip']} seed={j['seed']} total={rec['total']} "
                      f"avg={avg:.1f}s/video eta={avg * (len(todo) - n) / 3600:.1f}h parse_fail={n_bad} "
                      f"peak_mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
    print(f"done: {n} scored, {n_bad} parse failures -> {out}")


if __name__ == "__main__":
    main()
