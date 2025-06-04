import torch
import numpy as np
import cv2  # 需要事先安装 opencv-python

class RemoveSmallRegionsMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK", {}), 
                "min_area": ("INT", {"default": 50, "min": 0, "max": 1000000, "step": 1}),  # 最小保留面积
            }
        }

    RETURN_TYPES = ("MASK",)  # 输出是 ComfyUI Mask 类型
    RETURN_NAMES = ("cleaned_mask",)
    FUNCTION = "remove_regions"
    CATEGORY = "CozyMantis/Mask"  # 可以放在你的 Mask 分类下

    def remove_regions(self, mask, min_area):
        """
        mask: ComfyUI Mask 张量，形状一般为 (B, H, W) 或 (B, 1, H, W)，float32，值在 [0, 1] 之间
        min_area: 要保留的连通区域最小像素数量
        """

        # --- 一、取出 numpy，并保证是单通道 2D 数组 ---
        # 如果 mask 在 GPU 上，先 detach() 并移动到 CPU
        mask_tensor = mask
        if isinstance(mask_tensor, torch.Tensor):
            mask_tensor = mask_tensor.detach().cpu()

        # 假设 batch_size B >= 1，只处理第一张
        if mask_tensor.shape[0] > 1:
            print("Warning: Batch size > 1 detected. Only the first mask will be processed.")

        # 取第一张
        mask_np = mask_tensor[0].numpy()  # 这一步可能得到 shape=(H,W) 或 (1,H,W) 或 (H,W,1)

        # 把可能的“通道”或“批次”维度 squeeze 掉，确保是一个 (H, W) 的 2D 数组
        mask_np = np.squeeze(mask_np)
        # 此时 mask_np.ndim 应该为 2
        if mask_np.ndim != 2:
            raise ValueError(f"After squeeze, mask_np 维度应为 2D，但得到 {mask_np.shape}")

        # --- 二、阈值并转换为 uint8 二值图 ---
        # 先把浮点数值放大到 [0,255] 再二值化，确保得到 0 或 255 的 uint8
        # 这里可以用 cv2.threshold，也可以用 numpy 逻辑：
        # 方法一：numpy 方式
        binary_mask = ( (mask_np > 0.5).astype(np.uint8) ) * 255

        # 或者，你也可以用 OpenCV 的阈值接口（效果相同，只是写法不同）：
        # thresh_input = (mask_np * 255).astype(np.uint8)
        # _, binary_mask = cv2.threshold(thresh_input, 127, 255, cv2.THRESH_BINARY)

        # 检查一下：
        if binary_mask.dtype != np.uint8:
            raise TypeError(f"binary_mask dtype 应为 uint8，但得到 {binary_mask.dtype}")
        if binary_mask.ndim != 2:
            raise ValueError(f"binary_mask 维度应为 2，但得到 {binary_mask.shape}")
        if not (binary_mask.min() in [0] and binary_mask.max() in [0, 255]):
            raise ValueError(f"binary_mask 取值应为 0 或 255，但 min={binary_mask.min()} max={binary_mask.max()}")

        # --- 三、连通域分析并移除小连通域 ---
        # 使用 8 连通
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
            ltype=cv2.CV_32S
        )
        # labels: 与 binary_mask 同尺寸，每个像素标号 [0..num_labels-1]
        # stats: 形状为 (num_labels, 5)，每行 [x, y, width, height, area]
        # centroids: (num_labels, 2)

        # 新建一个空白图，用于放置保留的连通域
        cleaned_mask_np = np.zeros_like(binary_mask, dtype=np.uint8)

        # 从 1 开始遍历（跳过标签 0，即背景）
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask_np[labels == i] = 255

        # --- 四、把处理后结果转换回 ComfyUI 的 Mask 格式 (B, H, W), float32, [0,1] ---
        # 先归一化到 [0,1]
        final_mask_np = cleaned_mask_np.astype(np.float32) / 255.0  # 现在是一个 (H, W) 的 float32
        # 重新加回 batch 维度
        final_mask_tensor = torch.from_numpy(final_mask_np).unsqueeze(0)  # 变成 (1, H, W)

        return (final_mask_tensor, )
