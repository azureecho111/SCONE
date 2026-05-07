import os
import shutil
def find_first_uppercase(s):
    for i, char in enumerate(s):
        if char.isupper():
            return i, char
    return None, None
render = "/home1/yzh/icme2026/3d_front/datasets--huanngzh--3D-Front/snapshots/156fc5312ea613d4e92cbf5da1fddfa61bbad3b2/3D-FRONT-TEST-RENDER"
original_files = "./outputs/image2scene/test-3dfront-final-visualize/save"
temp_dir = "./curr/"
os.makedirs(temp_dir, exist_ok=True)

files = os.listdir(original_files)
files = [i for i in files if i.endswith("_scene.glb")]
for i in files:
    idx, char = find_first_uppercase(i)
    render_dir = os.path.join(render, i[13:idx-1] + "/" + i[idx:-10])
    os.makedirs(os.path.join(temp_dir, i[13:-10]), exist_ok=True)
    shutil.copy(os.path.join(original_files, i), os.path.join(os.path.join(temp_dir, i[13:-10]),"sketch.glb"))
    shutil.copy(os.path.join(render_dir, "semantic_0000.png"), os.path.join(os.path.join(temp_dir, i[13:-10]),"semantic_0000.png"))
    shutil.copy(os.path.join(render_dir, "render_0000.webp"), os.path.join(os.path.join(temp_dir, i[13:-10]),"render_0000.webp"))