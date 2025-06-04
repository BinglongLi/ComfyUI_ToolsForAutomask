import torch
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter # 需要安装 scipy

class DirectionalMaskExpansionCustomNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {}),
                "body_part": (["Torso", "Legs"], {}),
                "expansion_pixels": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather_pixels": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "openpose_keypoints": ("POSE_KEYPOINT", {}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("expanded_mask",)
    FUNCTION = "expand"
    CATEGORY = "CozyMantis/Mask"

    # Define OpenPose keypoint indices (based on COCO dataset, including MidHip)
    KEYPOINT_DICT = {
        "Neck": 1, "RShoulder": 2, "LShoulder": 5,
        "RHip": 8, "LHip": 11, "MidHip": 24,
        "RKnee": 9, "LKnee": 12,
        "RAnkle": 10, "LAnkle": 13,
    }
    MIN_CONFIDENCE = 0.3 # Minimum keypoint confidence threshold

    def get_keypoint_coords(self, keypoints_list, keypoint_name, canvas_width, canvas_height):
        """Safely get pixel coordinates for a keypoint."""
        index = self.KEYPOINT_DICT.get(keypoint_name)
        if index is None or not (0 <= index < len(keypoints_list)):
            return None, None, 0.0 # Index out of bounds or keypoints_list is too short

        x, y, conf = keypoints_list[index]

        if conf < self.MIN_CONFIDENCE:
             return None, None, conf # Confidence too low

        # Convert normalized coordinates (0-1) to pixel coordinates
        pixel_x = int(x * canvas_width)
        pixel_y = int(y * canvas_height)

        # Clamp coordinates to image bounds
        pixel_x = np.clip(pixel_x, 0, canvas_width - 1)
        pixel_y = np.clip(pixel_y, 0, canvas_height - 1)

        return pixel_x, pixel_y, conf

    def _get_torso_base_vector(self, keypoints_data_single_image, canvas_width, canvas_height):
        """
        Calculates the base pose vector (Neck/Shoulder towards Hip) representing the torso direction.
        Returns (dy, dx) vector and a reason string.
        This function only checks if the flat list length is a multiple of 3.
        """
        if not keypoints_data_single_image or 'people' not in keypoints_data_single_image or not keypoints_data_single_image['people']:
            return None, "NoPoseData"

        # Process first person
        person_data = keypoints_data_single_image['people'][0]

        if 'pose_keypoints_2d' not in person_data or not person_data['pose_keypoints_2d']:
             return None, "NoKeypoints2D"

        kps_flat = person_data['pose_keypoints_2d']
        # Only check if the flat list length is a multiple of 3
        if len(kps_flat) % 3 != 0:
            print(f"Warning: pose_keypoints_2d length {len(kps_flat)} is not a multiple of 3. Cannot process.")
            return None, "InvalidKeypointsLength"

        kps = np.array(kps_flat).reshape(-1, 3) # Shape: (NumKeypoints, 3)

        # --- Get keypoints for torso base vector calculation ---
        # get_keypoint_coords will return None if index is out of bounds or confidence is too low
        neck_x, neck_y, neck_conf = self.get_keypoint_coords(kps, "Neck", canvas_width, canvas_height)
        rs_x, rs_y, rs_conf = self.get_keypoint_coords(kps, "RShoulder", canvas_width, canvas_height)
        ls_x, ls_y, ls_conf = self.get_keypoint_coords(kps, "LShoulder", canvas_width, canvas_height)
        rh_x, rh_y, rh_conf = self.get_keypoint_coords(kps, "RHip", canvas_width, canvas_height)
        lh_x, lh_y, lh_conf = self.get_keypoint_coords(kps, "LHip", canvas_width, canvas_height)
        midhip_x, midhip_y, midhip_conf = self.get_keypoint_coords(kps, "MidHip", canvas_width, canvas_height)


        # --- Determine the Hip point ---
        hip_x, hip_y = None, None
        hip_confident = False
        if midhip_x is not None and midhip_y is not None:
             hip_x, hip_y = midhip_x, midhip_y
             hip_confident = True
        elif rh_x is not None and rh_y is not None and lh_x is not None and lh_y is not None:
             hip_x, hip_y = (rh_x + lh_x) // 2, (rh_y + lh_y) // 2
             hip_confident = True
        elif rh_x is not None and rh_y is not None:
             hip_x, hip_y = rh_x, rh_y
             hip_confident = True
        elif lh_x is not None and lh_y is not None:
             hip_x, hip_y = lh_x, lh_y
             hip_confident = True
        else:
             hip_confident = False


        # --- Determine the upper torso point (Neck or average Shoulder) ---
        upper_torso_x, upper_torso_y = None, None
        upper_torso_confident = False
        if neck_x is not None and neck_y is not None:
             upper_torso_x, upper_torso_y = neck_x, neck_y
             upper_torso_confident = True
        elif rs_x is not None and rs_y is not None and ls_x is not None and ls_y is not None:
             upper_torso_x, upper_torso_y = (rs_x + ls_x) // 2, (rs_y + ls_y) // 2
             upper_torso_confident = True
        elif rs_x is not None and rs_y is not None:
             upper_torso_x, upper_torso_y = rs_x, rs_y
             upper_torso_confident = True
        elif ls_x is not None and ls_y is not None:
             upper_torso_x, upper_torso_y = ls_x, ls_y
             upper_torso_confident = True
        else:
            upper_torso_confident = False


        if upper_torso_confident and hip_confident:
            # Vector from upper torso point towards hip point (downward direction)
            dy = hip_y - upper_torso_y
            dx = hip_x - upper_torso_x
            direction_vector = (dy, dx)

            # Normalize the vector
            dist = np.linalg.norm(direction_vector)
            if dist > 1e-6: # Avoid division by zero
                normalized_vector = (direction_vector[0] / dist, direction_vector[1] / dist)
                return normalized_vector, "TorsoBaseVectorCalculated"
            else:
                return (0, 0), "ZeroVectorFromBaseKeypoints" # Vector is effectively zero

        else:
             # Insufficient keypoints to calculate base vector
             return None, "InsufficientBaseKeypoints"


    def expand(self, mask, body_part, expansion_pixels, feather_pixels, openpose_keypoints=None):
        # Mask输入 (B, H, W), float32, 0-1
        mask_np = mask[0].numpy() # Shape: (H, W), float32, range: 0-1
        H, W = mask_np.shape

        # 将 Mask 转换为二值 numpy 数组 (0或255)
        binary_mask = (mask_np > 0.5).astype(np.uint8) * 255 # Shape: (H, W), dtype: uint8, values: 0 or 255

        # Initialize expanded_mask_np with the binary mask.
        expanded_mask_np = binary_mask.copy()

        # Only perform expansion logic if expansion_pixels is greater than 0
        if expansion_pixels > 0:
            effective_direction_vector = (0, 0) # Default to no movement
            reason = "NoExpansion" # Default reason

            # 1. Get the base pose vector (Torso direction) if OpenPose data is available
            base_vector = None
            base_reason = None
            if openpose_keypoints is not None and isinstance(openpose_keypoints, list) and len(openpose_keypoints) > 0:
                 try:
                     # Assume processing the first image in the batch
                     data_for_first_image = openpose_keypoints[0]
                     # _get_torso_base_vector returns normalized vector or (0,0) or None
                     base_vector, base_reason = self._get_torso_base_vector(data_for_first_image, W, H)
                 except IndexError:
                      base_reason = "EmptyPoseList"
                 except Exception as e:
                     base_reason = f"ProcessingError: {e}"
            else:
                 base_reason = "NoOpenPoseInputOrInvalidType"


            # 2. Determine the effective expansion vector based on body_part and base_vector
            # Only process if a base vector was successfully calculated AND is not effectively zero
            if base_vector is not None and np.linalg.norm(base_vector) > 1e-6:
                 if body_part == "Torso":
                      effective_direction_vector = base_vector # Use base vector (Neck/Shoulder -> Hip)
                      reason = "PoseGuidedTorso"
                 elif body_part == "Legs": # !!! 修改这里，处理 LegS 选项 !!!
                      # Use the opposite of the base vector (Hip -> Neck/Shoulder)
                      effective_direction_vector = (-base_vector[0], -base_vector[1])
                      reason = "PoseGuidedLegs"
                 else:
                     # This branch should ideally not be reached due to COMBO choices,
                     # but handle defensively.
                     print(f"Warning: Unhandled body_part '{body_part}' selected. No expansion.")
                     effective_direction_vector = (0, 0)
                     reason = f"UnhandledBodyPart({body_part})"
            else:
                 # Pose guidance failed (base_vector is None or effectively zero)
                 print(f"Warning: Pose guidance failed ({base_reason}). No directional expansion for '{body_part}'.")
                 effective_direction_vector = (0, 0) # Ensure vector is zero if pose fails
                 reason = f"PoseGuidanceFailed({base_reason if base_reason else 'NoBaseVector'})"


            # --- Debugging Prints ---
            print(f"Selected Body Part: {body_part}")
            print(f"Base Pose Vector Status: {base_reason}")
            if base_vector is not None:
                 print(f"Calculated Base Pose Vector (dy, dx): ({base_vector[0]:.4f}, {base_vector[1]:.4f})")
            print(f"Effective Expansion Vector (dy, dx): ({effective_direction_vector[0]:.4f}, {effective_direction_vector[1]:.4f})")
            print(f"Expansion Reason: {reason}")
            # ------------------------


            # 3. Perform directional expansion using the determined effective_direction_vector
            # Skip shifting if vector is zero
            if np.linalg.norm(effective_direction_vector) > 1e-6:
                 dy, dx = effective_direction_vector
                 # expanded_mask_np was already initialized with binary_mask.copy()

                 # Perform expansion step by step by shifting the ORIGINAL mask
                 # and accumulating the results using np.maximum
                 for i in range(expansion_pixels):
                     shifted_mask = np.zeros_like(binary_mask)

                     # Calculate the total shift for this step (i+1 pixels in the normalized direction)
                     total_shift_y = int(round(dy * (i + 1)))
                     total_shift_x = int(round(dx * (i + 1)))

                     # Calculate source and destination slices for the shift
                     src_slice_y_start = max(0, -total_shift_y)
                     src_slice_y_end = H - max(0, total_shift_y)
                     src_slice_x_start = max(0, -total_shift_x)
                     src_slice_x_end = W - max(0, total_shift_x)

                     dest_slice_y_start = max(0, total_shift_y)
                     dest_slice_y_end = H - max(0, -total_shift_y)
                     dest_slice_x_start = max(0, total_shift_x)
                     dest_slice_x_end = W - max(0, -total_shift_x)

                     src_slice_y = slice(src_slice_y_start, src_slice_y_end)
                     src_slice_x = slice(src_slice_x_start, src_slice_x_end)
                     dest_slice_y = slice(dest_slice_y_start, dest_slice_y_end)
                     dest_slice_x = slice(dest_slice_x_start, dest_slice_x_end)

                     if src_slice_y.start < src_slice_y.stop and src_slice_x.start < src_slice_x.stop and \
                        dest_slice_y.start < dest_slice_y.stop and dest_slice_x.start < dest_slice_x.stop:

                         shifted_mask[dest_slice_y, dest_slice_x] = binary_mask[src_slice_y, src_slice_x]

                     expanded_mask_np = np.maximum(expanded_mask_np, shifted_mask)

            else:
                 print(f"Note: No directional expansion applied for {body_part} due to zero vector or pose failure.")


        # ----------------------------------------------------------
        # 羽化 Mask (Feathering)
        # ----------------------------------------------------------
        feathered_mask_np = expanded_mask_np.astype(np.float32)

        if feather_pixels > 0:
            sigma = max(0.5, feather_pixels / 3.0)

            if sigma > 0:
                feathered_mask_np = gaussian_filter(feathered_mask_np, sigma=sigma)

            feathered_mask_np = np.clip(feathered_mask_np, 0, 255)

        # ----------------------------------------------------------
        # 准备输出
        # ----------------------------------------------------------
        final_mask_np = feathered_mask_np / 255.0
        final_mask_tensor = torch.from_numpy(final_mask_np).unsqueeze(0)

        return (final_mask_tensor,)

# NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS would be here