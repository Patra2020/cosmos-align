# VLM-Guided Flow-DPO for a Video World Model

Post-training **NVIDIA Cosmos3-Edge (4B)** to generate more physically consistent robot-manipulation
videos, using preferences produced by a **Qwen3.8-27B vision-language judge** instead of human labels.

> **Research question.** Can VLM-guided preference optimization (Flow-DPO) improve a pretrained
> video world model on robot manipulation, without any supervised fine-tuning stage?

| Component | Choice |
|---|---|
| Video generator | [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) (4B, image-to-video) |
| VLM judge | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (loaded 4-bit) |
| Dataset | [nvidia/BridgeData2-Subset-Synthetic-Captions](https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions) |
| Training | Flow-DPO from [NVIDIA cosmos-framework](https://github.com/NVIDIA/cosmos-framework) (`posttrain.entrypoints.dpo`), LoRA on the generation pathway |
| Hardware | RTX 4090 (24 GB) GPUs |

---

## Pipeline

```mermaid
flowchart LR
    subgraph data[BridgeData2 subset]
        T[1,222 train clips<br/>first frame + caption JSON]
        V[51 val clips<br/>held out]
    end

    subgraph rollout[1. Rollouts]
        G[Cosmos3-Edge base<br/>image-to-video]
        T --> G
        G -->|4 seeds per clip| C[~4,900 candidate videos]
    end

    subgraph judge[2. VLM judge]
        Q[Qwen3.8-27B]
        C --> Q
        Q -->|4 scores + rationale| S[Scored candidates]
    end

    subgraph pairs[3. Preference pairs]
        S -->|best vs worst<br/>same first frame| P[winner / loser JSONL<br/>filtered by score gap]
    end

    subgraph dpo[4. Flow-DPO]
        R[Frozen reference<br/>= base Edge]
        P --> D[Policy: Edge + LoRA]
        R -.-> D
        D --> M[Preference-optimized<br/>Cosmos3-Edge]
    end

    subgraph eval[5. Evaluation]
        V --> E[Base vs DPO<br/>human + VLM + physics checks]
        M --> E
        G --> E
    end
```

### Why skip SFT

The project plan originally had an SFT stage. It is skipped here on purpose: the question is whether
*preferences alone* can move a pretrained world model, and skipping SFT keeps the frozen reference equal
to the public checkpoint, which makes the comparison clean. SFT can be added back later as an ablation.

### How a preference pair is scored (Flow-DPO)

```mermaid
flowchart TB
    subgraph pair[One preference pair, same prompt and same first frame]
        W[winner video] --> VAE1[Wan 2.2 VAE encode]
        L[loser video] --> VAE2[Wan 2.2 VAE encode]
    end
    VAE1 --> N[Shared noise and timestep<br/>frame 0 kept clean - i2v conditioning]
    VAE2 --> N
    N --> PW[policy error on winner]
    N --> RW[reference error on winner]
    N --> PL[policy error on loser]
    N --> RL[reference error on loser]
    PW & RW & PL & RL --> LOSS["loss = -log σ( -β/2 · [ (e_pol_w − e_ref_w) − (e_pol_l − e_ref_l) ] )"]
```

The policy is pushed to denoise the winner better than the reference does, and the loser worse, relative
to the frozen reference. Frame 0 is a conditioning frame: it stays clean and is excluded from the loss,
so training matches image-to-video inference.

### Judge rubric

Each candidate is scored 1–5 on four axes, with a short written rationale per axis:

| Axis | What the judge looks for |
|---|---|
| Task completion | Does the video show the instruction in the caption being carried out? |
| Object interaction | Does the gripper actually contact and move the right object? |
| Physical consistency | No objects vanishing, teleporting, passing through each other, or melting |
| Temporal coherence | Stable appearance of arm, objects and background across frames |

Judge reliability is measured against a small set of independently annotated clips before the scores are
used for training (planned, see Status).

---

## Repository layout

```
cosmos-align/
├── README.md
├── requirements.txt              # inference / rollout / judge env
├── scripts/
│   └── infer_i2v_val.py          # single-clip image-to-video inference (baseline)
├── outputs/                      # generated videos + metadata JSON (gitignored except samples/)
├── BridgeData2/                  # dataset download (gitignored)
├── models/                       # Cosmos3-Edge, Qwen3.8-27B weights (gitignored)
└── packages/cosmos-framework/    # NVIDIA post-training framework, pinned commit (gitignored, see Setup)
```

---

## Setup

Two environments are needed because NVIDIA's post-training stack pins an older Torch/Transformers than
the Diffusers Cosmos3 pipeline and the Qwen judge require.

### 1. Inference, rollouts, judge (`cosmos` env)

```bash
conda create -n cosmos python=3.12 -y && conda activate cosmos
pip install -r requirements.txt
```

### 2. Weights and data

```bash
# accept the OpenMDW licence on the model page first, then
hf download nvidia/Cosmos3-Edge --local-dir models/Cosmos3-Edge
hf download Qwen/Qwen3.8-27B   --local-dir models/Qwen3.8-27B
hf download nvidia/BridgeData2-Subset-Synthetic-Captions --repo-type dataset --local-dir BridgeData2
```

### 3. Post-training env (Flow-DPO only)

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos-framework
cd packages/cosmos-framework && git checkout 709ecd9
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-posttrain"
uv sync --frozen --no-default-groups
uv sync --project docker/posttrain/environment --frozen --group cu130-runtime \
        --active --inexact --python .venv-posttrain/bin/python
```

---

## Usage

### Baseline inference on one held-out clip

```bash
conda activate cosmos
CUDA_VISIBLE_DEVICES=0 python scripts/infer_i2v_val.py --clip episode_002345_clip000 --seed 0
```

Reads the clip's upsampled JSON prompt, first frame, resolution, frame count and fps from the dataset's
`inference_prompt_i2v` file, runs the model card's Diffusers recipe, and writes the MP4 plus a JSON with
timing and peak memory to `outputs/base_i2v/`.

---

## Results

### Baseline (Cosmos3-Edge, no fine-tuning)

Measured on one RTX 4090, 256×256, 93 frames at 5 fps, 20 denoising steps, guidance 6.0:

| Metric | Value |
|---|---|
| Generation time | 12.1 s |
| Peak GPU memory | 9.0 GB |
| Model load (cold, NFS) | 75 s |

Qualitatively, the base model keeps the scene and arm stable but tends to **drop objects from the scene**:
in the bead-sweeping clip below most beads vanish within the first few frames and the sweep never gathers
them, while the real video keeps every bead and pushes them into a pile. This object-disappearance /
task-completion failure is exactly what the preference stage targets.

Top: generated. Bottom: ground-truth clip.

![baseline comparison](outputs/samples/episode_002345_clip000_base_vs_ref.png)

### Base vs. Flow-DPO

_Pending. Will report task-completion rate, physical-consistency violation rate, temporal coherence,
human preference win rate with confidence intervals, and VLM–human agreement of the judge._

---

## Status

- [x] Dataset downloaded and verified (1,222 train / 51 val, all splits complete)
- [x] Cosmos3-Edge baseline inference working on a single 24 GB GPU
- [x] Flow-DPO path in cosmos-framework audited: video pairs via JSONL, LoRA support, Edge→DCP converter, i2v conditioning via a custom sequence plan
- [ ] Rollout generation: 4 seeds × 1,222 train first frames
- [ ] Qwen3.8-27B judge (4-bit) with structured 4-axis scores
- [ ] Judge reliability check against independent labels
- [ ] Preference pair construction and filtering
- [ ] Flow-DPO training (Edge model config, LoRA, frozen base reference)
- [ ] Held-out evaluation and ablations (score margin, candidates per prompt, global vs pairwise judging)

---

## Notes and caveats

- **Visual task completion is not robot success.** No claim about policy performance is made unless the
  videos are later tied to executable actions and evaluated in simulation or on hardware.
- The dataset's prompts specify **5 fps**, below the 12–30 fps range listed on the model card. The pipeline
  accepts it, and matching the dataset's native rate was preferred over the documented range.
- The safety checker is disabled during research inference to avoid downloading the guardrail models.

## Acknowledgements

Cosmos3-Edge, the BridgeData2 subset and the cosmos-framework are released by NVIDIA under OpenMDW.
BridgeData V2 is from Walke et al., CoRL 2023. Qwen3.8 is released by the Qwen team under Apache 2.0.
