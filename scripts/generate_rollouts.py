"""Stage 1: generate N candidate videos per prompt with the base Cosmos3-Edge model.

For every clip in a BridgeData2 split, this script
  1. takes the clip's first frame (extracted from the MP4 for train, or the shipped JPEG for val),
  2. uses the clip's structured caption JSON as the image-to-video prompt,
  3. generates `--seeds` videos that differ only in the random seed,
  4. appends one JSON line per video to a manifest that later stages (judge, pairing, DPO) read.

Outputs (under --out-dir, default outputs/rollouts/<split>):
  first_frames/<clip>.jpg          conditioning image used for every seed of that clip
  prompts/<clip>.json              the exact prompt JSON string sent to the model
  videos/<clip>_seed<k>.mp4        generated candidates
  manifest.jsonl                   one line per video: clip, seed, paths, size, timing

The script is resumable: videos already present in the manifest are skipped. Use --shard i/n to split
the clip list across several processes/GPUs.

Examples:
  # smoke test
  CUDA_VISIBLE_DEVICES=1 python scripts/generate_rollouts.py --split train --limit 2 --seeds 0,1
  # full train run
  CUDA_VISIBLE_DEVICES=1 python scripts/generate_rollouts.py --split train
  # held-out val set for evaluation
  CUDA_VISIBLE_DEVICES=1 python scripts/generate_rollouts.py --split val
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Cosmos3-Edge"
DATA_DIR = ROOT / "BridgeData2" / "sft_dataset_bridge"


# --------------------------------------------------------------------------- data

def load_train_items(data_dir: Path) -> list[dict]:
    """One item per train clip: id, source video, prompt JSON (as str), native frame count and fps."""
    items = []
    with open(data_dir / "train" / "video_dataset_file.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            w = e["t2w_windows"][0]
            items.append(
                {
                    "clip": e["uuid"],
                    "video": str(data_dir / "train" / e["vision_path"]),
                    "image": None,
                    "prompt": json.dumps(w["caption_json"]),
                    "caption": w["caption"],
                    "native_frames": int(w["end_frame"]) - int(w["start_frame"]),
                    "fps": int(w["caption_json"].get("fps", 5)),
                    "resolution": int(w["caption_json"]["resolution"]["H"]),
                    "aspect_ratio": w["caption_json"].get("aspect_ratio", "1,1"),
                }
            )
    return items


def load_val_items(data_dir: Path) -> list[dict]:
    """One item per val clip, from the shipped i2v prompt files and first-frame JPEGs."""
    items = []
    pdir = data_dir / "val" / "inference_prompt_i2v"
    for p in sorted(pdir.glob("*.json")):
        spec = json.loads(p.read_text())
        clip = p.stem
        items.append(
            {
                "clip": clip,
                "video": str(data_dir / "val" / "videos" / f"{clip}.mp4"),
                "image": str((pdir / spec["vision_path"]).resolve()),
                "prompt": spec["prompt"],
                "caption": (data_dir / "val" / "captions" / clip / "caption.txt").read_text().strip(),
                "native_frames": int(spec["num_frames"]),
                "fps": int(spec["fps"]),
                "resolution": int(spec["resolution"]),
                "aspect_ratio": spec["aspect_ratio"],
            }
        )
    return items


def first_frame(video_path: str) -> Image.Image:
    import av

    with av.open(video_path) as c:
        for frame in c.decode(video=0):
            return frame.to_image().convert("RGB")
    raise RuntimeError(f"no frames decoded from {video_path}")


def size_from_spec(resolution: int, aspect_ratio: str) -> tuple[int, int]:
    """(width, height) with `resolution` as the short side, snapped to multiples of 16."""
    ar_w, ar_h = (int(x) for x in aspect_ratio.split(","))
    if ar_w >= ar_h:
        h, w = resolution, resolution * ar_w // ar_h
    else:
        w, h = resolution, resolution * ar_h // ar_w
    return w - w % 16, h - h % 16


def read_done(manifest: Path) -> set[tuple[str, int]]:
    done = set()
    if manifest.exists():
        with open(manifest) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if Path(r["video"]).exists():
                        done.add((r["clip"], int(r["seed"])))
    return done


# --------------------------------------------------------------------------- model

def load_pipeline(model_dir: Path, flow_shift: float):
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    pipe = Cosmos3OmniPipeline.from_pretrained(str(model_dir), dtype=torch.bfloat16, enable_safety_checker=False)
    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=flow_shift, use_karras_sigmas=False
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: outputs/rollouts/<split>")
    ap.add_argument("--seeds", default="0,1,2,3", help="comma-separated seeds; one video per seed per clip")
    ap.add_argument("--num-frames", type=int, default=93, help="frames per video (4k+1; model chunk is 93)")
    ap.add_argument("--fps", type=float, default=None, help="override fps (default: from the dataset prompt, 5)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--flow-shift", type=float, default=12.0, help="UniPC flow_shift from the model card recipe")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (smoke tests)")
    ap.add_argument("--clips", default=None, help="comma-separated clip ids to run instead of the whole split")
    ap.add_argument("--shard", default="0/1", help="i/n: process clips with index %% n == i")
    args = ap.parse_args()

    out_dir = args.out_dir or (ROOT / "outputs" / "rollouts" / args.split)
    for sub in ("first_frames", "prompts", "videos"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"

    items = load_train_items(DATA_DIR) if args.split == "train" else load_val_items(DATA_DIR)
    if args.clips:
        wanted = set(args.clips.split(","))
        items = [it for it in items if it["clip"] in wanted]
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    items = [it for k, it in enumerate(items) if k % shard_n == shard_i]
    if args.limit:
        items = items[: args.limit]
    seeds = [int(s) for s in args.seeds.split(",")]
    done = read_done(manifest)
    todo = [(it, s) for it in items for s in seeds if (it["clip"], s) not in done]
    print(f"split={args.split} clips={len(items)} seeds={seeds} already_done={len(done)} to_generate={len(todo)}")
    if not todo:
        return

    negative_prompt = json.dumps(json.loads((MODEL_DIR / "assets" / "negative_prompt.json").read_text()))

    t0 = time.time()
    pipe = load_pipeline(MODEL_DIR, args.flow_shift)
    print(f"model loaded in {time.time() - t0:.0f}s")

    from diffusers.utils import export_to_video

    n_ok = 0
    t_start = time.time()
    with open(manifest, "a") as mf:
        for it, seed in todo:
            clip = it["clip"]
            img_path = out_dir / "first_frames" / f"{clip}.jpg"
            if not img_path.exists():
                img = Image.open(it["image"]).convert("RGB") if it["image"] else first_frame(it["video"])
                img.save(img_path, quality=95)
            prompt_path = out_dir / "prompts" / f"{clip}.json"
            if not prompt_path.exists():
                prompt_path.write_text(json.dumps({"clip": clip, "prompt": it["prompt"], "caption": it["caption"]}))

            width, height = size_from_spec(it["resolution"], it["aspect_ratio"])
            fps = args.fps or float(it["fps"])
            video_path = out_dir / "videos" / f"{clip}_seed{seed}.mp4"

            t1 = time.time()
            result = pipe(
                prompt=it["prompt"],
                negative_prompt=negative_prompt,
                image=Image.open(img_path).convert("RGB"),
                num_frames=args.num_frames,
                height=height,
                width=width,
                fps=fps,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                add_resolution_template=False,
                add_duration_template=False,
            )
            export_to_video(result.video, str(video_path), fps=int(round(fps)), macro_block_size=1)
            gen_s = time.time() - t1

            rec = {
                "split": args.split,
                "clip": clip,
                "seed": seed,
                "video": str(video_path),
                "first_frame": str(img_path),
                "prompt_file": str(prompt_path),
                "reference_video": it["video"],
                "width": width,
                "height": height,
                "num_frames": args.num_frames,
                "fps": fps,
                "steps": args.steps,
                "guidance": args.guidance,
                "flow_shift": args.flow_shift,
                "generation_seconds": round(gen_s, 2),
            }
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            n_ok += 1

            if n_ok % 10 == 0 or n_ok == len(todo):
                elapsed = time.time() - t_start
                rate = elapsed / n_ok
                eta_h = rate * (len(todo) - n_ok) / 3600
                print(
                    f"[{n_ok}/{len(todo)}] {clip} seed={seed} {gen_s:.1f}s "
                    f"avg={rate:.1f}s/video eta={eta_h:.1f}h "
                    f"peak_mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
                    flush=True,
                )

    print(f"done: {n_ok} videos in {(time.time() - t_start) / 60:.1f} min -> {manifest}")


if __name__ == "__main__":
    main()
