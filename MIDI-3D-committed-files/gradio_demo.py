import os
import random
import uuid
from typing import Any, List, Optional, Union

import gradio as gr
import numpy as np
import torch
import trimesh
from gradio_image_prompter import ImagePrompter
from huggingface_hub import snapshot_download
from PIL import Image

from midi.pipelines.pipeline_midi import MIDIPipeline
# --- ADDED FOR SKETCH ADAPTER ---
from midi.sketch.fusion_adapter import FusionAdapterConfig, SketchFusionAdapter
from midi.sketch.sketch_tower import SketchVisionTowerConfig
from midi.models.transformers import TripoSGDiTModel
# --------------------------------
from scripts.grounding_sam import detect, plot_segmentation, prepare_model, segment
from scripts.image_to_textured_scene import (
    prepare_ig2mv_pipeline,
    prepare_texture_pipeline,
    run_i2tex,
)
from scripts.inference_midi import run_midi
from transformers import AutoProcessor
# Sketch Generator
from informative_drawings.model import Generator
from midi.sketch.sketch_utils import get_sketch_spatial_gating_map, prepare_sketch_images, tensor_to_pil_list

sketch_vision_tower_path = "openai/clip-vit-large-patch14"


# import spaces

# Constants
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "VAST-AI/MIDI-3D"

MARKDOWN = """
## Image to 3D Scene with [MIDI-3D](https://huanngzh.github.io/MIDI-Page/)
1. Upload an image, and draw bounding boxes for each instance by holding and dragging the mouse, or use text labels to segment the image. Then click "Run Segmentation" to generate the segmentation result. <b>Nota that if you select "box" mode, ensure instances should not be too small and bounding boxes fit snugly around each instance.</b>
2. <b>Check "Do image padding" in "Generation Settings" if instances in your image are too close to the image border.</b> Then click "Run Generation" to generate a 3D scene from the image and segmentation result.
3. If you find the generated 3D scene satisfactory, download it by clicking the "Download GLB" button.
"""

sketch_generator = Generator(3, 1, 3)
sketch_generator.load_state_dict(
    torch.load(os.path.join("checkpoints", "anime_style", 'netG_A_%s.pth' % "latest")))
sketch_generator.to(DEVICE).eval()
sketch_processor = AutoProcessor.from_pretrained(sketch_vision_tower_path)


EXAMPLES = [
    [
        {
            "image": "assets/example_data/Cartoon-Style/00_rgb.png",
        },
        "assets/example_data/Cartoon-Style/00_seg.png",
        42,
        False,
        False,
    ],
    [
        {
            "image": "assets/example_data/Cartoon-Style/01_rgb.png",
        },
        "assets/example_data/Cartoon-Style/01_seg.png",
        42,
        False,
        False,
    ],
    [
        {
            "image": "assets/example_data/Cartoon-Style/03_rgb.png",
        },
        "assets/example_data/Cartoon-Style/03_seg.png",
        42,
        False,
        False,
    ],
    [
        {
            "image": "assets/example_data/Realistic-Style/00_rgb.png",
        },
        "assets/example_data/Realistic-Style/00_seg.png",
        42,
        False,
        True,
    ],
    [
        {
            "image": "assets/example_data/Realistic-Style/01_rgb.png",
        },
        "assets/example_data/Realistic-Style/01_seg.png",
        42,
        False,
        True,
    ],
    [
        {
            "image": "assets/example_data/Realistic-Style/02_rgb.png",
        },
        "assets/example_data/Realistic-Style/02_seg.png",
        42,
        False,
        False,
    ],
    [
        {
            "image": "assets/example_data/Realistic-Style/05_rgb.png",
        },
        "assets/example_data/Realistic-Style/05_seg.png",
        42,
        False,
        False,
    ],
]

os.makedirs(TMP_DIR, exist_ok=True)

# Prepare models
## Grounding SAM
object_detector, sam_processor, sam_segmentator = prepare_model(
    device=DEVICE,
    detector_id="IDEA-Research/grounding-dino-tiny",
    segmenter_id="facebook/sam-vit-base",
)

## MIDI-3D
local_dir = "pretrained_weights/MIDI-3D"
# snapshot_download(local_dir=local_dir)

# --- ORIGINAL CODE (COMMENTED OUT) ---
# pipe: MIDIPipeline = MIDIPipeline.from_pretrained(local_dir).to(DEVICE, DTYPE)
# pipe.init_custom_adapter(
#     set_self_attn_module_names=[
#         "blocks.8",
#         "blocks.9",
#         "blocks.10",
#         "blocks.11",
#         "blocks.12",
#     ]
# )
# -------------------------------------

# --- ADDED FOR SKETCH ADAPTER ---
checkpoint_path = os.environ.get("SCONE_CKPT_PATH", "path/to/your/scone.ckpt")

sketch_fusion_adapter_config = FusionAdapterConfig(
    num_dit_layers=21,
    num_vision_layers=12,
    dim_dit_latent=2048,
    dim_vision_latent=1024,
    patch_size=14,
    fusion_mode="specific",
    dit_layer_seqs=[0,2,4,5,6,8,9,10,12,14,16,18,20],
    vision_layer_seqs=[0,1,2,3,4,5,6,7,8,9,10,11,12],
    enable_projector=False,
    dim_projected_latent=768
)

sketch_vision_tower_config = SketchVisionTowerConfig(
    select_feature_type="patch",
    arbitrary_input_size=False,
    input_size=(224,224),
    select_layer="all",
    dim_vision_latent=1024,
)

transformer = TripoSGDiTModel.from_pretrained(
    os.path.join(local_dir, "transformer"),
    strict=False,
    ignore_mismatched_sizes=True,
    low_cpu_mem_usage=False,
    dit_fusion_layer_seq=sketch_fusion_adapter_config.dit_layer_seqs,
    sketch_attention_dim=1024
).to("cpu")

sketch_fusion_adapter = SketchFusionAdapter(
    sketch_fusion_adapter_config,
    sketch_vision_tower_config
)

pipe: MIDIPipeline = MIDIPipeline.from_pretrained(
    local_dir,
    transformer=transformer,
    sketch_fusion_adapter=sketch_fusion_adapter,
    strict=False,
)

cross_attn_sketch_names = [f"blocks.{i}" for i in range(22)]

transformer_lora_config = {
    "r": 64,
    "lora_alpha": 32,
    "init_lora_weights": True,
    "target_modules": ["to_k", "to_q", "to_v", "to_out.0"]
}
image_encoder_2_lora_config = {
    "r": 64,
    "lora_alpha": 32,
    "init_lora_weights": True,
    "target_modules": ["projection", "attention.key", "attention.query", "attention.value", "output.dense", "mlp.fc1", "mlp.fc2"]
}
sketch_vision_tower_lora_config = {
    "r": 64,
    "lora_alpha": 32,
    "init_lora_weights": True,
    "target_modules": ["projection", "attention.key", "attention.query", "attention.value", "output.dense", "mlp.fc1", "mlp.fc2"]
}

pipe.init_custom_adapter(
    set_self_attn_module_names=["blocks.8", "blocks.9", "blocks.10", "blocks.11", "blocks.12"],
    set_sketch_attn_module_names=cross_attn_sketch_names,
    pretrained_image_encoder_2_processor_config=None,
    image_encoder_2_input_channels=7,
    image_encoder_2_init_projection_method="clone",
    transformer_lora_config=transformer_lora_config,
    image_encoder_1_lora_config=None, # image_encoder_1 does not use LoRA in your config
    image_encoder_2_lora_config=image_encoder_2_lora_config,
    sketch_vision_tower_lora_config=sketch_vision_tower_lora_config,
    load_ckpt=True
)

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        pipe.load_state_dict(checkpoint['state_dict'])
    else:
        pipe.load_state_dict(checkpoint)
else:
    print(f"Warning: Checkpoint {checkpoint_path} not found.")

pipe = pipe.to(DEVICE, DTYPE)
# --------------------------------

## MV-Adapter
# ig2mv_pipe = prepare_ig2mv_pipeline(device="cuda", dtype=torch.float16)
# texture_pipe = prepare_texture_pipeline(device="cuda", dtype=torch.float16)


@torch.no_grad()
# @torch.autocast(device_type=DEVICE, dtype=torch.float16)
def run_segmentation(
    image_prompts: Any,
    seg_mode: str,
    text_labels: Optional[str] = None,
    polygon_refinement: bool = True,
    detect_threshold: float = 0.3,
) -> Image.Image:
    rgb_image = image_prompts["image"].convert("RGB")

    segment_kwargs = {}
    if seg_mode == "box":
        # pre-process the layers and get the xyxy boxes of each layer
        if len(image_prompts["points"]) == 0:
            gr.Warning("Please draw bounding boxes for each instance on the image.")
            return None

        boxes = [
            [
                [int(box[0]), int(box[1]), int(box[3]), int(box[4])]
                for box in image_prompts["points"]
            ]
        ]

        if len(boxes) == 0 or any(len(box) == 0 for box in boxes):
            gr.Warning("Please draw bounding boxes for each instance on the image.")
            return None

        segment_kwargs["boxes"] = [boxes]
    else:
        if text_labels is None or text_labels == "" or len(text_labels.split(",")) == 0:
            gr.Warning("Please enter text labels separated by commas.")
            return None

        text_labels = text_labels.split(",")
        detections = detect(object_detector, rgb_image, text_labels, detect_threshold)
        segment_kwargs["detection_results"] = detections

    # run the segmentation
    detections = segment(
        sam_processor,
        sam_segmentator,
        rgb_image,
        polygon_refinement=polygon_refinement,
        **segment_kwargs,
    )
    seg_map_pil = plot_segmentation(rgb_image, detections)

    torch.cuda.empty_cache()

    return seg_map_pil


@torch.no_grad()
# @torch.autocast(device_type=DEVICE, dtype=torch.float16)
def run_generation(
    rgb_image: Any,
    seg_image: Union[str, Image.Image],
    seed: int,
    randomize_seed: bool = False,
    num_inference_steps: int = 35,
    guidance_scale: float = 7.0,
    do_image_padding: bool = False,
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    if not isinstance(rgb_image, Image.Image) and "image" in rgb_image:
        rgb_image = rgb_image["image"]

    # Prepare Sketch Images:
    rgb_tensor = torch.from_numpy(np.array(rgb_image.convert("RGB"))).float() / 255.0
    sketch_image = sketch_generator(rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE))
    sketch_image = sketch_image.repeat_interleave(3, dim=1).squeeze()
    sketch_image = tensor_to_pil_list(sketch_image.detach().cpu(), False)[0]
    # sketch_image = sketch_image.repeat_interleave(3,dim=1).squeeze()
    # sketch_image = tensor_to_pil_list(sketch_image.detach(), False)[0]

    scene = run_midi(
        pipe,
        rgb_image,
        seg_image,
        [sketch_image],
        seed,
        num_inference_steps,
        guidance_scale,
        do_image_padding,
    )

    # create uuid for the output file
    output_path = os.path.join(TMP_DIR, f"midi3d_{uuid.uuid4()}.glb")
    scene.export(output_path)

    torch.cuda.empty_cache()

    return output_path, output_path, seed


# @torch.no_grad()
# def apply_texture(scene_path: str, rgb_image: Any, seg_image: Any, seed: int):
#     if not isinstance(rgb_image, Image.Image) and "image" in rgb_image:
#         rgb_image = rgb_image["image"]

#     scene = trimesh.load(scene_path, process=False)
#     print(f"Loaded scene with {len(scene.geometry)} meshes")

#     # create a tmp dir
#     tmp_dir = os.path.join(TMP_DIR, f"textured_{uuid.uuid4()}")
#     os.makedirs(tmp_dir, exist_ok=True)
#     print(f"Created temporary directory: {tmp_dir}")

#     print("Starting texture generation process...")
#     textured_scene = run_i2tex(
#         ig2mv_pipe,
#         texture_pipe,
#         scene,
#         rgb_image,
#         seg_image,
#         seed,
#         output_dir=tmp_dir,
#     )
#     print(
#         f"Texture generation completed. Final scene has {len(textured_scene.geometry)} meshes"
#     )

#     output_path = os.path.join(tmp_dir, "textured_scene.glb")
#     textured_scene.export(output_path)
#     print(f"Exported textured scene to {output_path}")

#     torch.cuda.empty_cache()

#     return output_path, output_path, seed


# Demo
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                image_prompts = ImagePrompter(label="Input Image", type="pil")
                seg_image = gr.Image(
                    label="Segmentation Result", type="pil", format="png"
                )

            with gr.Accordion("Segmentation Settings", open=True):
                segmentation_mode = gr.Dropdown(
                    ["box", "label"],
                    value="box",
                    label="Segmentation Mode",
                    info="Box: Draw bounding boxes on the image to generate the segmentation result.\nLabel: Use text labels to segment the image.",
                )
                text_labels = gr.Textbox(
                    label="Text Labels",
                    value="",
                    placeholder="Enter text labels separated by commas if label mode is selected",
                )
                polygon_refinement = gr.Checkbox(label="Polygon Refinement", value=True)
            seg_button = gr.Button("Run Segmentation")

            with gr.Accordion("Generation Settings", open=False):
                do_image_padding = gr.Checkbox(label="Do image padding", value=False)
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=35,
                )
                guidance_scale = gr.Slider(
                    label="CFG scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=7.0,
                )
            gen_button = gr.Button("Run Generation", variant="primary")
            tex_button = gr.Button("Apply Texture", interactive=False)

        with gr.Column():
            model_output = gr.Model3D(label="Generated GLB", interactive=False)
            download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
            textured_model_output = gr.Model3D(label="Textured GLB", interactive=False)
            download_textured_glb = gr.DownloadButton(
                label="Download Textured GLB", interactive=False
            )

    with gr.Row():
        gr.Examples(
            examples=EXAMPLES,
            fn=run_generation,
            inputs=[image_prompts, seg_image, seed, randomize_seed, do_image_padding],
            outputs=[model_output, download_glb, seed],
            cache_examples=False,
        )

    seg_button.click(
        run_segmentation,
        inputs=[
            image_prompts,
            segmentation_mode,
            text_labels,
            polygon_refinement,
        ],
        outputs=[seg_image],
    ).then(lambda: gr.Button(interactive=True), outputs=[gen_button])

    gen_button.click(
        run_generation,
        inputs=[
            image_prompts,
            seg_image,
            seed,
            randomize_seed,
            num_inference_steps,
            guidance_scale,
            do_image_padding,
        ],
        outputs=[model_output, download_glb, seed],
    ).then(lambda: gr.Button(interactive=True), outputs=[download_glb]).then(
        lambda: gr.Button(interactive=True), outputs=[tex_button]
    )

    # tex_button.click(
    #     apply_texture,
    #     inputs=[model_output, image_prompts, seg_image, seed],
    #     outputs=[textured_model_output, download_textured_glb, seed],
    # ).then(lambda: gr.Button(interactive=True), outputs=[download_textured_glb])

demo.launch(server_name="0.0.0.0", server_port=7861)
