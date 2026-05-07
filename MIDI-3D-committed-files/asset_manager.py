import os
import json
import uuid
import shutil
from PIL import Image

# 资产库默认保存在与当前文件平级的 assets_library 目录中
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets_library")
METADATA_FILE = os.path.join(ASSETS_DIR, "metadata.json")

def init_library():
    """初始化资产库文件夹和JSON数据结构"""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'w') as f:
            json.dump([], f)

def save_asset(glb_path, thumbnail_image, name="未命名资产", tags=""):
    """
    保存单个 3D 资产及其缩略图到库中。
    :param glb_path: 刚生成的 3D 模型临时文件路径
    :param thumbnail_image: PIL.Image 格式的缩略图（通常可复用原图或分割图）
    :param name: 资产名称
    :param tags: 资产标签，如 "现代,沙发"
    """
    if not glb_path or not os.path.exists(glb_path):
        raise ValueError("无效的 GLB 模型路径")
        
    init_library()
    
    # 生成唯一ID以防止文件重名
    asset_id = str(uuid.uuid4())[:8]
    
    # 拷贝 3D 模型文件
    new_glb_filename = f"{asset_id}.glb"
    new_glb_path = os.path.join(ASSETS_DIR, new_glb_filename)
    shutil.copy2(glb_path, new_glb_path)
    
    # 处理并保存缩略图
    new_img_filename = f"{asset_id}.png"
    new_img_path = os.path.join(ASSETS_DIR, new_img_filename)
    if isinstance(thumbnail_image, Image.Image):
        thumbnail_image.save(new_img_path)
    elif isinstance(thumbnail_image, str) and os.path.exists(thumbnail_image):
        shutil.copy2(thumbnail_image, new_img_path)
    else:
        # 若实在没有缩略图，造一个纯黑图片占位
        img = Image.new('RGB', (256, 256), color = 'black')
        img.save(new_img_path)
        
    # 解析逗号分隔的标签
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    
    # 构造这条记录
    asset_record = {
        "id": asset_id,
        "name": name,
        "tags": tag_list,
        "glb_path": new_glb_filename,
        "thumbnail_path": new_img_filename,
    }
    
    # 写入 JSON 数据库
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    data.insert(0, asset_record) # 插入到最前面，以展示最新的
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    return asset_record

def load_all_assets():
    """提取全部库里的资产记录供前端展示"""
    init_library()
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_asset_absolute_path(filename):
    """获取资产文件的绝对路径，给 Gradio Model3D 提供正确的渲染路径"""
    return os.path.join(ASSETS_DIR, filename)

if __name__ == "__main__":
    init_library()
    print(f"Asset Library initialized at: {ASSETS_DIR}")
