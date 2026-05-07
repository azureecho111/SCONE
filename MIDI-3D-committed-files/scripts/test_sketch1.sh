#!/bin/bash

# 设置错误处理：遇到错误时执行清理函数
set -e

# 清理函数
cleanup_cuda() {
    echo "清理 CUDA 进程..."
    # 杀死可能的残留 Python 进程
    pkill -f "python.*launch.py" || true
    # 如果有僵尸进程，也可以尝试清理
    nvidia-smi --gpu-reset || true
    # 等待一下确保进程被清理
    sleep 2
}

# 设置陷阱，在脚本退出时执行清理（包括因错误退出）
trap cleanup_cuda EXIT

# 命令数组
commands=(
    "python launch.py --config configs/train/training_sketch_module_lr_search.yaml --wandb --train --gpu 4,5 --lr 1e-5"
    "python launch.py --config configs/train/training_sketch_module_lr_search.yaml --wandb --train --gpu 4,5 --lr 5e-5"
    "python launch.py --config configs/train/training_sketch_module_lr_search.yaml --wandb --train --gpu 4,5 --lr 1e-4"
    "python launch.py --config configs/train/training_sketch_module_lr_search.yaml --wandb --train --gpu 4,5 --lr 3e-4"
    "python launch.py --config configs/train/training_sketch_module_lr_search.yaml --wandb --train --gpu 4,5 --lr 5e-4"
    "python launch.py --config configs/train/training_sketch_module_lr_search.yaml --wandb --train --gpu 4,5 --lr 6e-4"
    "python launch.py --config configs/train/training_sketch_module_lr_search.yaml --wandb --train --gpu 4,5 --lr 7e-4"
    "python launch.py --config configs/train/training_sketch_module_lr_search.yaml --wandb --train --gpu 4,5 --lr 8e-4"
    "python launch.py --config configs/train/training_sketch_module_lr_search.yaml --wandb --train --gpu 4,5 --lr 9e-4"
    "python launch.py --config configs/train/training_sketch_module_lr_search.yaml --wandb --train --gpu 4,5 --lr 1e-3"
)

# 逐行执行命令
for cmd in "${commands[@]}"; do
    echo "执行命令: $cmd"
    if $cmd; then
        echo "命令执行成功"
    else
        echo "命令执行失败，退出码: $?"
        echo "执行清理并继续下一个命令..."
        cleanup_cuda
    fi
    echo "----------------------------------------"
done

echo "所有命令执行完毕"