import torch
import numpy as np
import cv2  # 需要先安装 opencv-python

# ======================================================================
# PruneThinBranchesMaskNode
# 对一张 mask（float32 [0,1]）先腐蚀再膨胀，用来剪掉“宽度 < (2*erode_size+1)”的所有细线分支
# ======================================================================
class PruneThinBranchesMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK", {}), 
                # erode_size: 腐蚀半径 (像素)，整张 mask 会被整体腐蚀 erode_size 次。 
                # 如果某条“线”宽度 ≤ 2*erode_size+1，就会被彻底抹掉，不会在后续的膨胀中恢复。
                "erode_size": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("pruned_mask",)
    FUNCTION = "prune_branches"
    CATEGORY = "CozyMantis/Mask"

    def prune_branches(self, mask, erode_size):
        """
        对 mask 进行：腐蚀 erode_size → 膨胀 erode_size，从而去掉所有“宽度 ≤ 2*erode_size+1”的细枝。

        参数：
          - mask:       ComfyUI Mask 张量，shape = (B, H, W)，float32，值在 [0,1]。
          - erode_size: 正整数，腐蚀迭代次数（等效半径）。若某个分支的最窄处 ≤ 2*erode_size+1，第一步腐蚀就会把它整条抹干净。

        返回：
          - pruned_mask: 同样是 (B, H, W)，float32 [0,1]。主体轮廓会近似恢复到原来大小，但所有窄分支消失了。
        """

        # 目前只处理第一张（batch size = 1）。若 B>1，仅提示并使用第一个元素。
        if mask.shape[0] > 1:
            print("Warning: batch size > 1, only process the first mask in the batch.")
        # 把 tensor 从 GPU 转到 CPU、detach，再变成 NumPy
        m_t = mask[0].detach().cpu()
        m_np = m_t.numpy()  # 可能 shape=(H,W) 或 (1,H,W)
        m_np = np.squeeze(m_np)  # 确保是 (H, W)
        if m_np.ndim != 2:
            raise ValueError(f"输入的 mask squeeze 后不是 2D，而是 {m_np.shape}")

        # 1) 二值化：把 [0,1] 化为 0/255 的 uint8
        #    使用阈值 0.5，把任何大于 0.5 的像素视为前景(255)，否则 0。
        #    这样可以把原来带灰度的抗锯齿边缘也地毯式清理成“干净的二值”。
        _, binary = cv2.threshold((m_np * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

        # 2) 先腐蚀（iterations = erode_size）
        #    kernel 选一个正方形（也可以选圆形）：我们使用 3×3 的矩形核，迭代 erode_size 次。
        #    如果某条线的最窄处 ≤ 2*erode_size+1，那么它将在这一步直接被整条“侵蚀掉”。
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(binary, kernel, iterations=erode_size)

        # 3) 再做同样次数的膨胀（iterations = erode_size）
        #    这样会让“剩下的粗壮主体”再恢复回和原来大小差不多（往外膨胀 erode_size 次），
        #    但原来被侵蚀掉的细线就不会再出现了。
        pruned = cv2.dilate(eroded, kernel, iterations=erode_size)

        # 4) （可选）去除孤立的小噪点
        #    经过腐蚀→膨胀后，主体一般都恢复得很好，
        #    但如果中间有孤立的 1 像素“残渣”也会被一起膨胀回来。可以选择性再做一次小开运算把它们去掉：
        #kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        #pruned = cv2.morphologyEx(pruned, cv2.MORPH_OPEN, kernel2, iterations=1)

        # 5) 转成 float32 [0,1] 并加回 batch 维度
        pruned_f = (pruned.astype(np.float32) / 255.0)  # shape = (H, W)
        pruned_t = torch.from_numpy(pruned_f).unsqueeze(0)  # 变成 (1, H, W)

        return (pruned_t,)
