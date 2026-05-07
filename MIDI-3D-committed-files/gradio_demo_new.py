import os
import random
import uuid
import time
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
from openai import OpenAI

from scripts.inference_midi import run_midi
from transformers import AutoProcessor
from asset_manager import save_asset, load_all_assets, get_asset_absolute_path
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
## Sketch/Image to 3D Scene with [MIDI-3D](https://huanngzh.github.io/MIDI-Page/)
**【比赛特别版】支持从草图生成RGB再生成3D！**
0. **[新增]** 如果你只有草图，可以在 "Step 0" 中上传或手绘草图，输入描述后点击 `生成RGB图` 调用豆包/大模型。
1. 在生成好的 RGB 图上（或自己上传RGB），拖拽鼠标为每个实例绘制边界框，也可以使用文本标签进行分割。然后单击 `Run Segmentation` 生成分割结果。<b>注意如果选择框选模式，边界框应紧贴实例。</b>
2. <b>如果图像中的实例太靠近图像边界，请在生成设置中选中“Do image padding”。</b>然后单击 `Run Generation` 生成3D场景。
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        pipe.load_state_dict(checkpoint['state_dict'])
    else:
        pipe.load_state_dict(checkpoint)
else:
    print(f"Warning: Checkpoint {checkpoint_path} not found.")

pipe = pipe.to(DEVICE, DTYPE)
# --------------------------------

# -------------------------------------------------------------
# [新功能] 草图转RGB模块 (调用豆包等API)
# -------------------------------------------------------------
import requests
import base64
from io import BytesIO


def call_doubao_api_sketch_to_image(sketch_pil: Image.Image, prompt: str) -> Image.Image:
    """
    调用云雾API（豆包 Seedream 模型）进行草图转RGB生图
    """
    print(f"[API Call] 正在调用草图生图大模型，Prompt: '{prompt}'")

    # 1. 将输入的草图转为 Base64 格式
    buffered = BytesIO()
    # 确保图片为 RGB 格式
    sketch_pil.convert("RGB").save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 构建 Data URI (一般用于图生图参考格式传参)
    base64_image_url = f"data:image/jpeg;base64,{img_b64}"
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("Please set ARK_API_KEY before using sketch-to-RGB generation.")

    client = OpenAI(
        base_url=os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        api_key=api_key,
    )
    imagesResponse = client.images.generate(
        model="doubao-seedream-4-0-250828",
        prompt=prompt,
        size="1024x1024",
        response_format="url",
        extra_body={
            "image": base64_image_url,
            "watermark": True
        }
    )

    response = requests.get(imagesResponse.data[0].url)
    response.raise_for_status()  # 检查请求是否成功

    # 转换为Image对象
    image_data = BytesIO(response.content)
    img = Image.open(image_data).convert("RGB")
    return img
def on_generate_rgb(sketch_img: Optional[Image.Image], prompt: str):
    """
    前端点击“1. 草图生成RGB”时的响应函数
    """
    if sketch_img is None:
        raise gr.Error("请先在左侧上传或画一张草图！(Please provide a sketch first!)")
        
    generated_rgb = call_doubao_api_sketch_to_image(sketch_img, prompt)
    
    # 返回一个字典传递给 ImagePrompter 组件，它会把图展出来让人能框选
    return {"image": generated_rgb, "points": []}
# -------------------------------------------------------------

@torch.no_grad()
def run_segmentation(
    image_prompts: Any,
    seg_mode: str,
    text_labels: Optional[str] = None,
    polygon_refinement: bool = True,
    detect_threshold: float = 0.3,
) -> Image.Image:
    if image_prompts is None or "image" not in image_prompts:
        raise gr.Error("未找到输入的RGB图像！")
        
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


# Demo
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)

    # -------------------------------------------------------------
    # Step 0: 草图生图 模块
    # -------------------------------------------------------------
    with gr.Accordion("Step 0: Sketch to RGB (评委展示特别流程)", open=True):
        with gr.Row():
            with gr.Column():
                sketch_input = gr.Image(label="手绘/输入草图", interactive=True, type="pil")
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="RGB场景描述 Prompt (给大模型看)",
                    placeholder="e.g. A gorgeous anime bedroom with a wooden floor and desk...",
                    value="Highly detailed cartoon style 3D game scene."
                )
                gen_rgb_btn = gr.Button("1. 草图生成高质量RGB图", variant="primary")
                gr.Markdown("*点击后会调用大模型API(例如豆包/ControlNet)将上面输入的草图变为RGB图片并填入底下的主流程中。*")
    # -------------------------------------------------------------

    with gr.Row():
        with gr.Column():
            with gr.Row():
                # 这一步本来是上传图，现在可以由上面的模块自动填充
                image_prompts = ImagePrompter(label="主流程：原图 / 所生成的RGB图（请在此框选以分割）", type="pil")
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
            seg_button = gr.Button("2. Run Segmentation")

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
            gen_button = gr.Button("3. Run Generation", variant="primary")
            tex_button = gr.Button("Apply Texture", interactive=False)

        with gr.Column():
            model_output = gr.Model3D(label="Generated GLB", interactive=False)
            with gr.Row():
                download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
                save_to_lib_btn = gr.Button("⭐存入我的素材库", interactive=False)
                
            textured_model_output = gr.Model3D(label="Textured GLB", interactive=False)
            download_textured_glb = gr.DownloadButton(
                label="Download Textured GLB", interactive=False
            )

    # ------ [新功能] 个人 3D 资产库 ------
    with gr.Accordion("📂 我的高定私有素材库 (Personal 3D Library)", open=True):
        with gr.Row():
            with gr.Column(scale=2):
                asset_gallery = gr.Gallery(
                    label="资产列表 (点击预览)",
                    show_label=False,
                    object_fit="cover",
                    height=300,
                    interactive=False
                )
                refresh_btn = gr.Button("🔄 刷新库")
            with gr.Column(scale=1):
                library_model_view = gr.Model3D(label="资产检索预览室", interactive=False)
    # ------------------------------------

    with gr.Row():
        gr.Examples(
            examples=EXAMPLES,
            fn=run_generation,
            inputs=[image_prompts, seg_image, seed, randomize_seed, do_image_padding],
            outputs=[model_output, download_glb, seed],
            cache_examples=False,
        )

    # === 新增逻辑：刷新画廊数据及选中查看模型 ===
    def get_gallery_items():
        try:
            assets = load_all_assets()
            return [(get_asset_absolute_path(a["thumbnail_path"]), a["name"]) for a in assets]
        except Exception:
            return []
            
    asset_gallery.value = get_gallery_items()
        
    def on_gallery_select(evt: gr.SelectData):
        assets = load_all_assets()
        if evt.index < len(assets):
            return get_asset_absolute_path(assets[evt.index]["glb_path"])
        return None
        
    def on_save_to_library(prompts, glb_path):
        if not glb_path:
            gr.Warning("您还没生成模型！")
            return get_gallery_items()
            
        img = prompts["image"] if (isinstance(prompts, dict) and "image" in prompts) else prompts
        if img:
            save_asset(glb_path, img.convert("RGB"), name="自定义生成单品", tags="内部生成")
            gr.Info("⭐ 成功存入预制素材库！")
        else:
            gr.Warning("缺少缩略图。")
        return get_gallery_items()

        return get_gallery_items()

    refresh_btn.click(fn=get_gallery_items, inputs=[], outputs=[asset_gallery])
    asset_gallery.select(fn=on_gallery_select, inputs=[], outputs=[library_model_view])
    save_to_lib_btn.click(fn=on_save_to_library, inputs=[image_prompts, model_output], outputs=[asset_gallery])
    # ========================================

    # 绑定 草图 -> RGB 按钮事件
    gen_rgb_btn.click(
        fn=on_generate_rgb,
        inputs=[sketch_input, prompt_input],
        outputs=[image_prompts]
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
    ).then(lambda: gr.update(interactive=True), outputs=[gen_button])

    def enable_btns():
        return gr.update(interactive=True), gr.update(interactive=True)

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
    ).then(
        enable_btns, 
        outputs=[download_glb, save_to_lib_btn]
    ).then(
        lambda: gr.update(interactive=True), outputs=[tex_button]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862)
