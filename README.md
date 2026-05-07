<div align="center">

# SCONE: Sketch-Guided Spatial Gating for Multi-Object 3D Scene Reconstruction

[![Paper](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader)](./scone.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](./MIDI-3D-committed-files/LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)

</div>

---

**SCONE** (Sketch-Conditioned Object NEtwork) is the first end-to-end sketch-conditioned framework for single-image multi-object 3D scene reconstruction. Built on top of [MIDI-3D](https://huanngzh.github.io/MIDI-Page/), it introduces a novel **Spatial Gating Attention (SGA)** mechanism that fuses per-instance sketch geometry into DiT layers to sharpen object boundaries and improve geometric consistency — with **no additional manual annotation** required.

<!-- ![Teaser](docs/teaser.png) -->

## News

- **[2026-05]** Code and paper are released.

## Installation

```bash
git clone https://github.com/<your-name>/SCONE.git
cd SCONE/MIDI-3D-committed-files

conda create -n scone python=3.10 -y
conda activate scone

# Install PyTorch for your CUDA version: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Pretrained Weights

| Model | Source | Path |
|---|---|---|
| MIDI-3D base | Auto-downloaded from HuggingFace (`VAST-AI/MIDI-3D`) | `pretrained_weights/MIDI-3D/` |
| Informative Drawings sketch generator | [carolineec/informative-drawings](https://github.com/carolineec/informative-drawings) | `checkpoints/anime_style/netG_A_latest.pth` |
| SCONE checkpoint | Trained by yourself (see below) | Set via `checkpoint_path` in configs |

## Dataset

We train and evaluate on [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) following the same preprocessing protocol as MIDI-3D.

```bash
# Download test assets from HuggingFace (huanngzh/3D-Front)
python download_dataset.py
```

After downloading, update `scene_list`, `object_list`, `surface_root_dir`, `image_root_dir`, and `ckpt_path` in the YAML configs under `configs/train/` and `configs/test/`.

## Inference

```bash
cd MIDI-3D-committed-files

python scripts/inference_midi_final.py \
    --rgb   assets/example_data/Cartoon-Style/00_rgb.png \
    --seg   assets/example_data/Cartoon-Style/00_seg.png \
    --sketch /path/to/sketch.png \
    --seed 42 \
    --output-dir outputs/
```

The result is exported as `outputs/output.glb`.

## Gradio Demo

```bash
cd MIDI-3D-committed-files
python gradio_demo_new.py
# Visit http://localhost:7862
```

The demo supports uploading an RGB image, drawing bounding boxes for segmentation, and generating a 3D scene. An optional sketch-to-RGB step and a local 3D asset library are also included.

## Training

SCONE uses a two-stage training strategy.

**Stage 1** — Train SGA modules and sketch vision encoder with the diffusion backbone frozen:

```bash
cd MIDI-3D-committed-files

python launch.py \
    --config configs/train/training_sketch_module.yaml \
    --train --gpu 0,1,2,3
```

**Stage 2** — Full fine-tuning with LoRA:

```bash
python launch.py \
    --config configs/train/training_sketch_module_stage2.yaml \
    --train --gpu 0,1,2,3 --wandb
```

Checkpoints and logs are saved to `outputs/`.

## Evaluation

```bash
cd MIDI-3D-committed-files

python launch.py \
    --config configs/test/test_sketch_module_3dfront.yaml \
    --test --gpu 0
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{scone2026,
  title   = {SCONE: Sketch-Guided Spatial Gating for Multi-Object 3D Scene Reconstruction},
  author  = {},
  year    = {2026}
}
```

We also recommend citing [MIDI-3D](https://huanngzh.github.io/MIDI-Page/), which this work builds upon.

## Acknowledgements

This project builds on [MIDI-3D](https://github.com/VAST-AI-Research/MIDI-3D), [Informative Drawings](https://github.com/carolineec/informative-drawings), [Segment Anything](https://github.com/facebookresearch/segment-anything), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [MV-Adapter](https://github.com/huanngzh/mv-adapter). We thank the authors for releasing their code.

## License

This project is released under the [Apache 2.0 License](./MIDI-3D-committed-files/LICENSE). The `informative_drawings` module is under the [MIT License](./MIDI-3D-committed-files/informative_drawings/LICENSE).
