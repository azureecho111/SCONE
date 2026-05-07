import torch
import numpy as np


def analyze_single_tensor(tensor, name="tensor"):
    """详细分析单个tensor的各种属性"""
    print(f"\n{'=' * 60}")
    print(f"分析: {name}")
    print(f"{'=' * 60}")

    # 基本属性
    print(f"形状: {tensor.shape}")
    print(f"数据类型: {tensor.dtype}")
    print(f"设备: {tensor.device}")
    print(f"总元素数: {tensor.numel()}")
    print(f"存储大小: {tensor.element_size() * tensor.numel() / 1024:.2f} KB")
    print(f"是否需要梯度: {tensor.requires_grad}")

    # 检查是否为视图或连续
    print(f"是否连续: {tensor.is_contiguous()}")
    if hasattr(tensor, 'storage_offset'):
        print(f"存储偏移: {tensor.storage_offset()}")

    # NaN和Inf分析
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    nan_count = nan_mask.sum().item()
    inf_count = inf_mask.sum().item()
    total_count = tensor.numel()

    print(f"NaN数量: {nan_count}/{total_count} ({nan_count / total_count * 100:.2f}%)")
    print(f"Inf数量: {inf_count}/{total_count} ({inf_count / total_count * 100:.2f}%)")

    # 有效值分析
    valid_mask = ~nan_mask & ~inf_mask
    valid_count = valid_mask.sum().item()

    if valid_count > 0:
        valid_values = tensor[valid_mask]
        print(f"有效值数量: {valid_count}/{total_count} ({valid_count / total_count * 100:.2f}%)")
        print(f"有效值范围: [{valid_values.min().item():.10f}, {valid_values.max().item():.10f}]")
        print(f"有效值均值: {valid_values.mean().item():.10f}")
        print(f"有效值标准差: {valid_values.std().item():.10f}")
        print(f"有效值中位数: {valid_values.median().item():.10f}")

        # 分位数分析 - 修复类型问题
        if valid_count >= 5:
            # 确保使用浮点类型计算分位数
            if not valid_values.is_floating_point():
                valid_values_float = valid_values.float()
                print(f"⚠️ 转换为浮点类型计算分位数: {valid_values.dtype} -> {valid_values_float.dtype}")
            else:
                valid_values_float = valid_values

            # 确保分位数张量在正确设备上
            quantile_tensor = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9],
                                           device=valid_values_float.device,
                                           dtype=valid_values_float.dtype)

            try:
                quantiles = torch.quantile(valid_values_float, quantile_tensor)
                print(f"有效值分位数:")
                print(f"  10%: {quantiles[0].item():.10f}")
                print(f"  25%: {quantiles[1].item():.10f}")
                print(f"  50%: {quantiles[2].item():.10f}")
                print(f"  75%: {quantiles[3].item():.10f}")
                print(f"  90%: {quantiles[4].item():.10f}")
            except Exception as e:
                print(f"❌ 分位数计算失败: {e}")
    else:
        print("⚠️ 没有有效值!")

    # NaN分布分析
    if nan_count > 0:
        print("\nNaN分布分析:")
        # 检查每个维度的NaN分布
        if tensor.dim() > 0:
            for dim in range(tensor.dim()):
                nan_along_dim = nan_mask.sum(dim=dim)
                if nan_along_dim.dim() > 0:
                    unique_counts = torch.unique(nan_along_dim)
                    print(f"  维度{dim}: {unique_counts.tolist()}")

        # 找出NaN的位置模式
        nan_indices = torch.nonzero(nan_mask)
        if len(nan_indices) > 0:
            print(f"前5个NaN位置:")
            for i in range(min(5, len(nan_indices))):
                print(f"  位置 {nan_indices[i].tolist()}")

    # Inf分布分析
    if inf_count > 0:
        print("\nInf分布分析:")
        inf_indices = torch.nonzero(inf_mask)
        if len(inf_indices) > 0:
            print(f"前5个Inf位置:")
            for i in range(min(5, len(inf_indices))):
                idx = inf_indices[i]
                value = tensor[tuple(idx)]
                print(f"  位置 {idx.tolist()}: {value.item()}")

    # 零值分析
    zero_mask = tensor == 0
    zero_count = zero_mask.sum().item()
    print(f"零值数量: {zero_count}/{total_count} ({zero_count / total_count * 100:.2f}%)")

    # 极端值分析
    if valid_count > 0:
        abs_values = torch.abs(valid_values)
        large_values = abs_values > 1e6
        large_count = large_values.sum().item()
        small_values = abs_values < 1e-6
        small_count = small_values.sum().item()

        print(f"绝对值>1e6的数量: {large_count}/{valid_count} ({large_count / valid_count * 100:.2f}%)")
        print(f"绝对值<1e-6的数量: {small_count}/{valid_count} ({small_count / valid_count * 100:.2f}%)")

    # 内存布局分析
    print(f"\n内存布局:")
    print(f"步长: {tensor.stride()}")
    print(f"数据指针: {tensor.data_ptr()}")

    return {
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'device': tensor.device,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'valid_count': valid_count,
        'zero_count': zero_count,
    }


def compare_tensors(tensor1, tensor2, name="tensor"):
    """比较两个tensor的详细差异"""
    print(f"\n{'=' * 60}")
    print(f"比较: {name}")
    print(f"{'=' * 60}")

    # 基本属性比较
    print(f"形状: {tensor1.shape} vs {tensor2.shape}")
    print(f"数据类型: {tensor1.dtype} vs {tensor2.dtype}")
    print(f"设备: {tensor1.device} vs {tensor2.device}")

    # 检查形状是否匹配
    if tensor1.shape != tensor2.shape:
        print("❌ 形状不匹配!")
        return False

    # 数值比较
    diff = torch.abs(tensor1 - tensor2)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()

    print(f"最大绝对差异: {max_diff:.10f}")
    print(f"平均绝对差异: {mean_diff:.10f}")

    # 检查是否完全相同
    if torch.equal(tensor1, tensor2):
        print("✅ 两个tensor完全相同")
        return True
    else:
        print("❌ tensor存在差异")

        # 检查差异的分布
        zero_diff_mask = diff == 0
        zero_count = zero_diff_mask.sum().item()
        total_count = tensor1.numel()
        print(f"相同元素比例: {zero_count}/{total_count} ({zero_count / total_count * 100:.2f}%)")

        # 找出差异最大的位置
        if max_diff > 0:
            max_diff_idx = torch.argmax(diff)
            max_diff_idx_tuple = np.unravel_index(max_diff_idx.cpu().numpy(), tensor1.shape)
            print(f"最大差异位置: {max_diff_idx_tuple}")
            print(
                f"该位置值: {tensor1.flatten()[max_diff_idx].item():.10f} vs {tensor2.flatten()[max_diff_idx].item():.10f}")

        # 检查NaN和Inf
        nan_mask1 = torch.isnan(tensor1)
        nan_mask2 = torch.isnan(tensor2)
        inf_mask1 = torch.isinf(tensor1)
        inf_mask2 = torch.isinf(tensor2)

        print(f"tensor1 NaN数量: {nan_mask1.sum().item()}")
        print(f"tensor2 NaN数量: {nan_mask2.sum().item()}")
        print(f"tensor1 Inf数量: {inf_mask1.sum().item()}")
        print(f"tensor2 Inf数量: {inf_mask2.sum().item()}")

        # 检查NaN位置是否一致
        if nan_mask1.any() or nan_mask2.any():
            nan_positions_match = torch.equal(nan_mask1, nan_mask2)
            print(f"NaN位置是否一致: {nan_positions_match}")
            if not nan_positions_match:
                # 找出不一致的NaN位置
                nan_mismatch = (nan_mask1 != nan_mask2)
                mismatch_count = nan_mismatch.sum().item()
                print(f"NaN位置不一致数量: {mismatch_count}")

        return False


def detailed_nan_analysis(tensor, name="tensor"):
    """详细分析tensor中的NaN分布"""
    print(f"\n--- {name} NaN分析 ---")

    nan_mask = torch.isnan(tensor)
    nan_count = nan_mask.sum().item()
    total_count = tensor.numel()

    if nan_count == 0:
        print("✅ 没有NaN值")
        return

    print(f"NaN数量: {nan_count}/{total_count} ({nan_count / total_count * 100:.2f}%)")

    # 分析NaN的分布模式
    if tensor.dim() > 0:
        # 检查每个维度的NaN分布
        print("各维度的NaN分布:")
        for dim in range(tensor.dim()):
            nan_along_dim = nan_mask.sum(dim=dim)
            if nan_along_dim.dim() > 0:
                unique_counts = torch.unique(nan_along_dim)
                print(f"  维度{dim}: {unique_counts.tolist()}")

    # 检查有效值的统计
    valid_values = tensor[~nan_mask]
    if len(valid_values) > 0:
        print(f"有效值范围: [{valid_values.min().item():.6f}, {valid_values.max().item():.6f}]")
        print(f"有效值均值: {valid_values.mean().item():.6f}")
        print(f"有效值标准差: {valid_values.std().item():.6f}")


def compare_value_ranges(tensor1, tensor2, name="tensor"):
    """比较两个tensor的数值范围"""
    print(f"\n--- {name} 数值范围比较 ---")

    # 排除NaN后的统计
    tensor1_valid = tensor1[~torch.isnan(tensor1)]
    tensor2_valid = tensor2[~torch.isnan(tensor2)]

    if len(tensor1_valid) > 0 and len(tensor2_valid) > 0:
        print(f"tensor1范围: [{tensor1_valid.min().item():.10f}, {tensor1_valid.max().item():.10f}]")
        print(f"tensor2范围: [{tensor2_valid.min().item():.10f}, {tensor2_valid.max().item():.10f}]")
        print(f"tensor1均值: {tensor1_valid.mean().item():.10f}")
        print(f"tensor2均值: {tensor2_valid.mean().item():.10f}")
    else:
        print("⚠️ 无法计算有效值的统计（可能全是NaN）")


# 加载tensor
print("加载tensor文件...")
latent1 = torch.load('latents1-5.pth')  # 有病
latent2 = torch.load('latents1-2.pth')  # 没病
sampled_points1 = torch.load('sampled_points1-5.pth')
sampled_points2 = torch.load('sampled_points1-2.pth')
tensor1 = torch.load('tensor1-5.pth')
tensor2 = torch.load('tensor1-2.pth')

# =========================================
before_scheduler_latents = torch.load("before_scheduler_latents1-8.pth")
prepare_latents = torch.load("prepare_latents1-8.pth")
scheduler_latents = torch.load("scheduler_latents1-8.pth")
noise = torch.load("noise_pred1-8.pth")

before_scheduler_latents2 = torch.load("before_scheduler_latents1-9.pth")
prepare_latents2 = torch.load("prepare_latents1-9.pth")
scheduler_latents2 = torch.load("scheduler_latents1-9.pth")
noise2 = torch.load("noise_pred1-9.pth")
# 1-7 和 1-8 都是最后sdf为正数的failure case，但是他们结果完全相同。

print("✅ 所有tensor加载完成")

# 首先单独分析每个tensor
print("\n" + "=" * 80)
print("单独分析每个tensor")
print("=" * 80)

print('Flash Attention',torch.backends.cuda.sdp_kernel)

# analyze_single_tensor(latent1, "latent1 (有病)")
# analyze_single_tensor(latent2, "latent2 (没病)")
# analyze_single_tensor(sampled_points1, "sampled_points1")
# analyze_single_tensor(sampled_points2, "sampled_points2")
# analyze_single_tensor(tensor1, "tensor1 (有病)")
# analyze_single_tensor(tensor2, "tensor2 (没病)")
analyze_single_tensor(before_scheduler_latents, "before_scheduler")
analyze_single_tensor(prepare_latents, "prepare")
analyze_single_tensor(scheduler_latents, "scheduler")
analyze_single_tensor(noise, "noise")

compare_tensors(before_scheduler_latents, before_scheduler_latents2, "(COMPARE)before_scheduler")
compare_tensors(prepare_latents, prepare_latents2, "(COMPARE)prepare_latents")
compare_tensors(scheduler_latents, scheduler_latents2, "(COMPARE)scheduler")
compare_tensors(noise, noise2, "(COMPARE)noise")


# # 然后进行比较分析
# print("\n" + "=" * 80)
# print("比较分析")
# print("=" * 80)
#
# # 比较latents
# compare_tensors(latent1, latent2, "latents")
# detailed_nan_analysis(latent1, "latent1 (有病)")
# detailed_nan_analysis(latent2, "latent2 (没病)")
# compare_value_ranges(latent1, latent2, "latents")
#
# # 比较sampled_points
# compare_tensors(sampled_points1, sampled_points2, "sampled_points")
# detailed_nan_analysis(sampled_points1, "sampled_points1")
# detailed_nan_analysis(sampled_points2, "sampled_points2")
# compare_value_ranges(sampled_points1, sampled_points2, "sampled_points")
#
# # 比较输出tensor
# compare_tensors(tensor1, tensor2, "output tensor")
# detailed_nan_analysis(tensor1, "tensor1 (有病)")
# detailed_nan_analysis(tensor2, "tensor2 (没病)")
# compare_value_ranges(tensor1, tensor2, "output tensor")