"""Image-to-video inference with Cosmos3-Edge on one BridgeData2 val example.

Follows the Diffusers example in the Cosmos3-Edge model card. The val
inference_prompt_i2v/*.json files already hold the upsampled JSON prompt plus
resolution / aspect ratio / num_frames / fps and the first-frame image path.

Usage:
  python infer_i2v_val.py --clip episode_002345_clip000 [--seed 0] [--steps 20]
"""
import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

MODEL_DIR = Path("/data/cosmos-align/models/Cosmos3-Edge")
VAL_DIR = Path("/data/cosmos-align/BridgeData2/sft_dataset_bridge/val")
OUT_DIR = Path("/data/cosmos-align/outputs/base_i2v")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="episode_002345_clip000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--num-frames", type=int, default=None, help="override num_frames from the prompt file")
    ap.add_argument("--fps", type=float, default=None, help="override fps from the prompt file")
    args = ap.parse_args()

    spec = json.loads((VAL_DIR / "inference_prompt_i2v" / f"{args.clip}.json").read_text())
    negative_prompt = json.loads((MODEL_DIR / "assets" / "negative_prompt.json").read_text())
    image_path = (VAL_DIR / "inference_prompt_i2v" / spec["vision_path"]).resolve()

    res = int(spec["resolution"])
    ar_w, ar_h = (int(x) for x in spec["aspect_ratio"].split(","))
    # 'resolution' is the short side; derive the long side from the aspect ratio.
    if ar_w >= ar_h:
        height, width = res, res * ar_w // ar_h
    else:
        width, height = res, res * ar_h // ar_w
    width -= width % 16
    height -= height % 16
    num_frames = args.num_frames or int(spec["num_frames"])
    fps = args.fps or float(spec["fps"])

    print(f"clip={args.clip} image={image_path}")
    print(f"size={width}x{height} num_frames={num_frames} fps={fps} steps={args.steps} seed={args.seed}")

    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16, enable_safety_checker=False
    )
    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=12.0, use_karras_sigmas=False
    )
    print(f"model loaded in {time.time() - t0:.1f}s")

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    result = pipe(
        prompt=spec["prompt"],  # already a JSON string
        negative_prompt=json.dumps(negative_prompt),
        image=load_image(str(image_path)),
        num_frames=num_frames,
        height=height,
        width=width,
        fps=fps,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        add_resolution_template=False,
        add_duration_template=False,
    )
    gen_s = time.time() - t1
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.clip}_seed{args.seed}.mp4"
    export_to_video(result.video, str(out_path), fps=int(round(fps)), macro_block_size=1)

    meta = {
        "clip": args.clip, "seed": args.seed, "steps": args.steps, "guidance": args.guidance,
        "width": width, "height": height, "num_frames": num_frames, "fps": fps,
        "generation_seconds": round(gen_s, 1), "peak_gpu_gb": round(peak_gb, 2),
        "output": str(out_path), "reference_video": str(VAL_DIR / "videos" / f"{args.clip}.mp4"),
    }
    (OUT_DIR / f"{args.clip}_seed{args.seed}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
