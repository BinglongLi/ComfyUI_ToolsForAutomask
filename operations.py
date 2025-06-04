import torch
import numpy as np
import cv2  # 需要安装 opencv-python

# ======================================================================
# 1. 精准减法节点：SubtractMaskNode
#    将 mask_b 从 mask_a 中扣除（A & ~B），输出严格二值化后的结果
# ======================================================================
class PreciseSubtractMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_a": ("MASK", {}),
                "mask_b": ("MASK", {}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("subtracted_mask",)
    FUNCTION = "precise_subtract"
    CATEGORY = "CozyMantis/Mask"

    def precise_subtract(self, mask_a, mask_b):
        """
        mask_a, mask_b: ComfyUI Mask 张量，形状 (B, H, W)，float32，值在 [0,1]
        返回：从 mask_a 中扣除 mask_b 后的掩码 (B, H, W)，float32，值 0/1
        """
        # 只处理第一张（batch size 假设为 1）
        if mask_a.shape[0] > 1 or mask_b.shape[0] > 1:
            print("Warning: Batch size > 1 detected. Only the first mask of each will be processed.")
        a_np = mask_a[0].detach().cpu().numpy()
        b_np = mask_b[0].detach().cpu().numpy()

        # 去除多余维度，确保 a_np 和 b_np 都是 (H, W)
        a_np = np.squeeze(a_np)
        b_np = np.squeeze(b_np)
        if a_np.ndim != 2 or b_np.ndim != 2:
            raise ValueError(f"Inputs must be 2D after squeeze, but got {a_np.shape} and {b_np.shape}")

        # 二值化：阈值 0.5 → 0/255
        _, a_bin = cv2.threshold((a_np * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        _, b_bin = cv2.threshold((b_np * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

        # 精准减法：a_bin & (~b_bin)
        b_inv = cv2.bitwise_not(b_bin)
        sub = cv2.bitwise_and(a_bin, b_inv)

        # 转回 float32 [0,1] 并加上 batch 维度
        out_np = (sub.astype(np.float32) / 255.0)
        out_t = torch.from_numpy(out_np).unsqueeze(0)  # (1, H, W)
        return (out_t,)


# ======================================================================
# 2. 精准加法节点：AddMaskNode
#    将 mask_a 与 mask_b 作并集（A | B），输出严格二值化后的结果
# ======================================================================
class PreciseAddMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_a": ("MASK", {}),
                "mask_b": ("MASK", {}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("added_mask",)
    FUNCTION = "precise_add"
    CATEGORY = "CozyMantis/Mask"

    def precise_add(self, mask_a, mask_b):
        """
        mask_a, mask_b: ComfyUI Mask 张量，形状 (B, H, W)，float32，值在 [0,1]
        返回：mask_a 与 mask_b 的并集 (B, H, W)，float32，值 0/1
        """
        # 只处理第一张
        if mask_a.shape[0] > 1 or mask_b.shape[0] > 1:
            print("Warning: Batch size > 1 detected. Only the first mask of each will be processed.")
        a_np = mask_a[0].detach().cpu().numpy()
        b_np = mask_b[0].detach().cpu().numpy()

        # 去除多余维度
        a_np = np.squeeze(a_np)
        b_np = np.squeeze(b_np)
        if a_np.ndim != 2 or b_np.ndim != 2:
            raise ValueError(f"Inputs must be 2D after squeeze, but got {a_np.shape} and {b_np.shape}")

        # 二值化：阈值 0.5 → 0/255
        _, a_bin = cv2.threshold((a_np * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        _, b_bin = cv2.threshold((b_np * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

        # 精准加法：a_bin | b_bin
        added = cv2.bitwise_or(a_bin, b_bin)

        # 转回 float32 [0,1] 并加上 batch 维度
        out_np = (added.astype(np.float32) / 255.0)
        out_t = torch.from_numpy(out_np).unsqueeze(0)
        return (out_t,)


# ======================================================================
# 3. 闭运算节点：ClosingMaskNode
#    对单张 mask 做形态学闭运算（先膨胀再腐蚀），去除扣区小缝隙
# ======================================================================
class ClosingMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK", {}),
                # kernel_size 必须为奇数，建议 3 或 5
                "kernel_size": ("INT", {"default": 3, "min": 3, "max": 51, "step": 2}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("closed_mask",)
    FUNCTION = "closing"
    CATEGORY = "CozyMantis/Mask"

    def closing(self, mask, kernel_size):
        """
        mask: ComfyUI Mask 张量，形状 (B, H, W)，float32，值在 [0,1]
        kernel_size: 奇数，形态学运算核大小
        返回：对 mask 做形态学闭运算后 (B, H, W)，float32，值 0/1
        """
        # 只处理第一张
        if mask.shape[0] > 1:
            print("Warning: Batch size > 1 detected. Only the first mask will be processed.")
        m_np = mask[0].detach().cpu().numpy()

        # 去除多余维度
        m_np = np.squeeze(m_np)
        if m_np.ndim != 2:
            raise ValueError(f"Input mask must be 2D after squeeze, but got {m_np.shape}")

        # 二值化：阈值 0.5 → 0/255
        _, m_bin = cv2.threshold((m_np * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

        # 形态学闭运算：先膨胀再腐蚀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(m_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 转回 float32 [0,1] 并加上 batch 维度
        out_np = (closed.astype(np.float32) / 255.0)
        out_t = torch.from_numpy(out_np).unsqueeze(0)
        return (out_t,)


# ======================================================================
# 4. 开运算节点：OpeningMaskNode
#    对单张 mask 做形态学开运算（先腐蚀再膨胀），去除单像素噪点
# ======================================================================
class OpeningMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK", {}),
                # kernel_size 必须为奇数，建议 3 或 5
                "kernel_size": ("INT", {"default": 3, "min": 3, "max": 51, "step": 2}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("opened_mask",)
    FUNCTION = "opening"
    CATEGORY = "CozyMantis/Mask"

    def opening(self, mask, kernel_size):
        """
        mask: ComfyUI Mask 张量，形状 (B, H, W)，float32，值在 [0,1]
        kernel_size: 奇数，形态学运算核大小
        返回：对 mask 做形态学开运算后 (B, H, W)，float32，值 0/1
        """
        # 只处理第一张
        if mask.shape[0] > 1:
            print("Warning: Batch size > 1 detected. Only the first mask will be processed.")
        m_np = mask[0].detach().cpu().numpy()

        # 去除多余维度
        m_np = np.squeeze(m_np)
        if m_np.ndim != 2:
            raise ValueError(f"Input mask must be 2D after squeeze, but got {m_np.shape}")

        # 二值化：阈值 0.5 → 0/255
        _, m_bin = cv2.threshold((m_np * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

        # 形态学开运算：先腐蚀再膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(m_bin, cv2.MORPH_OPEN, kernel, iterations=1)

        # 转回 float32 [0,1] 并加上 batch 维度
        out_np = (opened.astype(np.float32) / 255.0)
        out_t = torch.from_numpy(out_np).unsqueeze(0)
        return (out_t,)


# ======================================================================
# 5. 条件选择：ConditionalMaskSelector
#    对两个mask选择，第一个非空选一，空则选二
# ======================================================================
class ConditionalMaskSelector:
    """
    Selects the first mask if it is not empty (contains non-zero pixels),
    otherwise selects the second mask.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK", {}), # The primary mask to check
                "mask2": ("MASK", {}), # The fallback mask if mask1 is empty
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("output_mask",)

    FUNCTION = "select_mask"

    CATEGORY = "Utils/Mask" # Or choose another appropriate category

    # Optional: Add a display name for the node
    # NODE_DISPLAY_NAME = "Conditional Mask Selector"

    def select_mask(self, mask1, mask2):
        """
        Checks if mask1 is empty and returns either mask1 or mask2.

        Args:
            mask1 (torch.Tensor): The primary mask (B, H, W, float32, 0-1).
            mask2 (torch.Tensor): The fallback mask (B, H, W, float32, 0-1).

        Returns:
            tuple: A tuple containing the selected mask (B, H, W, float32, 0-1).
        """
        # ComfyUI masks are torch tensors with shape (B, H, W) and dtype float32, range 0-1.
        # An empty mask would have all values close to 0.
        # We can check if the sum of the mask's elements is effectively zero.
        # Using sum() is generally robust for non-negative values.
        # Add a small epsilon for floating point comparisons.
        if torch.sum(mask1) > 10:
            # mask1 is not empty, return mask1
            print("Conditional Mask Selector: mask1 is not empty, selecting mask1.")
            return (mask1,)
        else:
            # mask1 is empty, return mask2
            print("Conditional Mask Selector: mask1 is empty, selecting mask2.")
            return (mask2,)

