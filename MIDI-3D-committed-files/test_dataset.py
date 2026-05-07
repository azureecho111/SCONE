import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os

def load_and_compare_glb_npy(glb_path, npy_path, num_samples=None, threshold=0.01, visualize=False):
    """
    完整的GLB和NPY文件等价性验证

    Args:
        glb_path: GLB文件路径
        npy_path: NPY文件路径
        num_samples: 从GLB采样的点数（如果为None，则使用NPY中的点数）
        threshold: 点云匹配的容差阈值
        visualize: 是否可视化比较结果
    """

    # 1. 诊断NPY文件结构
    print("=== NPY文件诊断 ===")
    npy_data = np.load(npy_path, allow_pickle=True)

    # 处理不同的NPY文件格式
    if npy_data.shape == ():  # 0维数组，包含字典
        npy_data = npy_data.item()
        print("NPY文件包含字典对象")

        if isinstance(npy_data, dict):
            print("字典键:", list(npy_data.keys()))

            # 查找点云数据
            point_keys = ['surface_points', 'points', 'vertices', 'point_cloud']
            npy_points = None
            for key in point_keys:
                if key in npy_data:
                    npy_points = npy_data[key]
                    print(f"找到点云数据: {key}, 形状: {npy_points.shape}")
                    break

            if npy_points is None:
                # 如果没有标准键，查找第一个二维数组
                for key, value in npy_data.items():
                    if hasattr(value, 'shape') and len(value.shape) == 2 and value.shape[1] == 3:
                        npy_points = value
                        print(f"找到可能的点云数据: {key}, 形状: {npy_points.shape}")
                        break

            if npy_points is None:
                raise ValueError("在NPY文件中未找到点云数据")
        else:
            raise ValueError("NPY文件包含的不是字典对象")
    else:
        # 直接是点云数组
        npy_points = npy_data
        print(f"NPY文件直接包含点云数组, 形状: {npy_points.shape}")

    # 2. 从GLB文件采样点云
    print("\n=== 从GLB文件采样点云 ===")
    try:
        mesh = trimesh.load(glb_path)

        # 处理场景文件
        if isinstance(mesh, trimesh.Scene):
            print("检测到场景文件，合并所有网格...")
            meshes = []
            for node_name in mesh.graph.nodes_geometry:
                transform, geometry_name = mesh.graph[node_name]
                geometry = mesh.geometry[geometry_name]
                if isinstance(geometry, trimesh.Trimesh):
                    transformed_mesh = geometry.copy()
                    transformed_mesh.apply_transform(transform)
                    meshes.append(transformed_mesh)

            if meshes:
                mesh = trimesh.util.concatenate(meshes)
                print(f"合并后网格: {len(mesh.vertices)} 个顶点, {len(mesh.faces)} 个面")
            else:
                raise ValueError("场景中没有找到有效的网格")

        # 确定采样点数
        if num_samples is None:
            num_samples = len(npy_points)
            print(f"使用NPY点数进行采样: {num_samples}")

        # 采样点云
        glb_points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
        print(f"GLB采样点云形状: {glb_points.shape}")

    except Exception as e:
        print(f"加载GLB文件失败: {e}")
        return False

    # 3. 点云预处理和归一化
    print("\n=== 点云预处理 ===")

    # 中心化
    npy_center = np.mean(npy_points, axis=0)
    glb_center = np.mean(glb_points, axis=0)

    npy_centered = npy_points - npy_center
    glb_centered = glb_points - glb_center

    # 缩放归一化
    npy_scale = np.max(np.linalg.norm(npy_centered, axis=1))
    glb_scale = np.max(np.linalg.norm(glb_centered, axis=1))

    if npy_scale > 0 and glb_scale > 0:
        npy_normalized = npy_centered / npy_scale
        glb_normalized = glb_centered / glb_scale
    else:
        npy_normalized = npy_centered
        glb_normalized = glb_centered

    print(f"NPY点云 - 中心: {npy_center}, 尺度: {npy_scale:.4f}")
    print(f"GLB点云 - 中心: {glb_center}, 尺度: {glb_scale:.4f}")

    # 4. 点云比较
    print("\n=== 点云比较 ===")

    # 使用KDTree进行最近邻匹配
    tree = cKDTree(glb_normalized)
    distances, indices = tree.query(npy_normalized, k=1)

    # 统计匹配结果
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    median_distance = np.median(distances)
    match_ratio = np.sum(distances < threshold) / len(distances)

    print(f"点云匹配结果:")
    print(f"  平均距离: {mean_distance:.6f}")
    print(f"  最大距离: {max_distance:.6f}")
    print(f"  中位数距离: {median_distance:.6f}")
    print(f"  匹配率 (距离<{threshold}): {match_ratio:.2%}")

    # 5. 可视化比较
    if visualize:
        visualize_comparison(npy_normalized, glb_normalized, distances, threshold,
                             npy_path, glb_path)

    # 6. 判断等价性
    is_equivalent = match_ratio > 0.95  # 匹配率超过95%认为等价

    print(f"\n=== 结论 ===")
    print(f"GLB文件和NPY文件是否等价: {is_equivalent}")

    if is_equivalent:
        print("✓ 文件很可能是等价的（NPY点云是从GLB采样得到的）")
    else:
        print("✗ 文件可能不等价，或者存在坐标变换差异")

    return is_equivalent, match_ratio, mean_distance


def visualize_comparison(npy_points, glb_points, distances, threshold, npy_path, glb_path):
    """可视化点云比较结果"""

    # 创建图形
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(f"GLB vs NPY 点云比较\n"
                 f"NPY: {os.path.basename(npy_path)}\n"
                 f"GLB: {os.path.basename(glb_path)}",
                 fontsize=14)

    # 子图1: NPY点云
    ax1 = fig.add_subplot(141, projection='3d')
    scatter1 = ax1.scatter(npy_points[:, 0], npy_points[:, 1], npy_points[:, 2],
                           c=distances, cmap='viridis', s=1, alpha=0.7)
    ax1.set_title('NPY点云\n(颜色表示到GLB的距离)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(scatter1, ax=ax1, shrink=0.6, label='距离')

    # 子图2: GLB采样点云
    ax2 = fig.add_subplot(142, projection='3d')
    scatter2 = ax2.scatter(glb_points[:, 0], glb_points[:, 1], glb_points[:, 2],
                           c='red', s=1, alpha=0.7)
    ax2.set_title('GLB采样点云')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # 子图3: 叠加显示
    ax3 = fig.add_subplot(143, projection='3d')
    scatter3_npy = ax3.scatter(npy_points[:, 0], npy_points[:, 1], npy_points[:, 2],
                               c='blue', s=1, alpha=0.5, label='NPY')
    scatter3_glb = ax3.scatter(glb_points[:, 0], glb_points[:, 1], glb_points[:, 2],
                               c='red', s=1, alpha=0.5, label='GLB')
    ax3.set_title('叠加显示')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()

    # 子图4: 距离分布直方图
    ax4 = fig.add_subplot(144)
    n, bins, patches = ax4.hist(distances, bins=50, alpha=0.7, color='green',
                                edgecolor='black')
    ax4.axvline(threshold, color='red', linestyle='--',
                label=f'阈值={threshold}', linewidth=2)
    ax4.set_xlabel('点对距离')
    ax4.set_ylabel('频数')
    ax4.set_title('点对距离分布')
    ax4.legend()

    # 添加统计信息
    stats_text = (f"平均距离: {np.mean(distances):.4f}\n"
                  f"最大距离: {np.max(distances):.4f}\n"
                  f"匹配率: {np.sum(distances < threshold) / len(distances):.2%}")
    ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()


def batch_compare_glb_npy(glb_dir, npy_dir, threshold=0.01):
    """
    批量比较GLB和NPY文件

    Args:
        glb_dir: 包含GLB文件的目录
        npy_dir: 包含NPY文件的目录
        threshold: 匹配阈值
    """

    # 获取文件列表
    glb_files = [f for f in os.listdir(glb_dir) if f.endswith('.glb')]
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]

    print(f"找到 {len(glb_files)} 个GLB文件, {len(npy_files)} 个NPY文件")

    results = []

    for glb_file in glb_files:
        base_name = os.path.splitext(glb_file)[0]
        npy_file = base_name + '.npy'

        if npy_file in npy_files:
            print(f"\n{'=' * 50}")
            print(f"比较文件: {base_name}")
            print(f"{'=' * 50}")

            glb_path = os.path.join(glb_dir, glb_file)
            npy_path = os.path.join(npy_dir, npy_file)

            try:
                is_equiv, match_ratio, mean_dist = load_and_compare_glb_npy(
                    glb_path, npy_path, threshold=threshold, visualize=False
                )

                results.append({
                    'file': base_name,
                    'equivalent': is_equiv,
                    'match_ratio': match_ratio,
                    'mean_distance': mean_dist
                })

            except Exception as e:
                print(f"比较失败: {e}")
                results.append({
                    'file': base_name,
                    'equivalent': False,
                    'match_ratio': 0,
                    'mean_distance': float('inf'),
                    'error': str(e)
                })
        else:
            print(f"未找到对应的NPY文件: {npy_file}")

    # 输出批量比较结果
    print(f"\n{'=' * 60}")
    print("批量比较结果摘要")
    print(f"{'=' * 60}")

    equiv_count = sum(1 for r in results if r.get('equivalent', False))
    print(f"等价文件数量: {equiv_count}/{len(results)} ({equiv_count / len(results) * 100:.1f}%)")

    if results:
        avg_match_ratio = np.mean([r['match_ratio'] for r in results if 'match_ratio' in r])
        avg_distance = np.mean([r['mean_distance'] for r in results if 'mean_distance' in r])
        print(f"平均匹配率: {avg_match_ratio:.2%}")
        print(f"平均距离: {avg_distance:.6f}")

    return results
# 使用示例
glb_file = "3d_front/datasets--huanngzh--3D-Front/snapshots/156fc5312ea613d4e92cbf5da1fddfa61bbad3b2/3D-FRONT-TEST-SCENE/deaa4ac0-8d13-423f-930f-78fb25825e8d/Bedroom-5088/Bed_69a2550c-ec04-44d9-a621-e601330a9cb6_1.glb"
npy_file = "3d_front/datasets--huanngzh--3D-Front/snapshots/156fc5312ea613d4e92cbf5da1fddfa61bbad3b2/3D-FRONT-TEST-SURFACE/deaa4ac0-8d13-423f-930f-78fb25825e8d/Bedroom-5088/Bed_69a2550c-ec04-44d9-a621-e601330a9cb6_1.npy"

is_equivalent, match_ratio, mean_distance = load_and_compare_glb_npy(
    glb_file, npy_file, threshold=0.01, visualize=False
)

print(is_equivalent)
print(f"Match ratio: {match_ratio}")
print(f"Mean Distance: {mean_distance}")