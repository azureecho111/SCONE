import os


def count_subfolders(path):
    """统计目录中的子文件夹数量"""
    if not os.path.exists(path):
        return 0

    count = 0
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            count += 1
    return count


# 使用示例
folder_path = "3d_front/datasets--huanngzh--3D-Front/snapshots/156fc5312ea613d4e92cbf5da1fddfa61bbad3b2/3D-FRONT-TEST-SURFACE"
num_folders = count_subfolders(folder_path)
print(f"子文件夹数量: {num_folders}")