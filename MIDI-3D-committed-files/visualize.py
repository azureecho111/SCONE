import trimesh
import pyrender
import numpy as np
from PIL import Image

glb_file = r"curr/THIS_dee82180-497b-489f-8554-cb27be17300f-Bedroom-46392/sketch.glb"

# =====================================================
# 1. Load scene
# =====================================================
tm_scene = trimesh.load(glb_file)
assert isinstance(tm_scene, trimesh.Scene)

# =====================================================
# 2. Scene 几何中心居中 + 归一化
# =====================================================
scene_center = tm_scene.centroid

for geom in tm_scene.geometry.values():
    geom.apply_translation(-scene_center)

bbox = tm_scene.bounds
radius = 0.5 * np.linalg.norm(bbox[1] - bbox[0])

# =====================================================
# 3. 创建 pyrender scene
# =====================================================
scene = pyrender.Scene(bg_color=[255, 255, 255, 255])

colors = [
    [180,  80, 200, 255],
    [ 80, 180, 200, 255],
    [200, 180,  80, 255],
    [120, 200, 120, 255],
    [200, 120, 120, 255],
]

for i, geom in enumerate(tm_scene.geometry.values()):
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=np.array(colors[i % len(colors)]) / 255.0,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        emissiveFactor=[0.05, 0.05, 0.05]
    )
    scene.add(pyrender.Mesh.from_trimesh(geom, material=material, smooth=False))

# =====================================================
# 4. 自动 fit 相机（关键）
# =====================================================
def look_at(eye, target, up=[0, 0, 1]):
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)

    z = eye - target
    z /= np.linalg.norm(z)

    x = np.cross(up, z)
    x /= np.linalg.norm(x)

    y = np.cross(z, x)

    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, y, z], axis=1)
    pose[:3, 3] = eye
    return pose

yfov = np.pi / 3
distance = radius / np.tan(yfov / 2) * 1.4

eye = [0, 0, distance]
cam_pose = look_at(
    eye=eye,
    target=[0, 0, 0],
    up=[0, 1, 0]   # 非 Z 轴，防止退化
)
camera = pyrender.PerspectiveCamera(yfov=yfov)
scene.add(camera, pose=cam_pose)

# =====================================================
# 5. 灯光（和相机一致）
# =====================================================
scene.add(pyrender.DirectionalLight(intensity=3.0), pose=cam_pose)
scene.add(pyrender.DirectionalLight(intensity=1.0), pose=np.eye(4))

# =====================================================
# 6. 高质量离屏渲染（SSAA）
# =====================================================
W, H = 2048, 2048
renderer = pyrender.OffscreenRenderer(W, H)

color, _ = renderer.render(scene)
renderer.delete()

img = Image.fromarray(color)
img = img.resize((1024, 1024), Image.LANCZOS)
img.save("result.png")

print("Saved result.png")