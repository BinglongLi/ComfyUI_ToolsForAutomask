import torch
import numpy as np
import cv2 # Requires opencv-python installed

class MaskFillGapsConvexHullNode:
    """
    Fills gaps in a mask based on its convex hull.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {}), # The input mask with potential gaps
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("filled_mask_hull",)

    FUNCTION = "fill_gaps_hull"

    CATEGORY = "CozyMantis/Mask"

    # Optional: Add a display name for the node
    # NODE_DISPLAY_NAME = "Mask Fill Gaps (Convex Hull)"

    def fill_gaps_hull(self, mask):
        mask_np = mask[0].numpy() # H, W, float32, 0-1

        # Convert to binary numpy array (uint8, 0 or 255) for OpenCV
        mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255 # Ensure 0 or 255

        H, W = mask_uint8.shape

        # Find the coordinates of all white pixels
        # np.where returns two arrays, one for rows (y) and one for columns (x)
        # np.column_stack combines them into a (N, 2) array of (x, y) points
        # Note: OpenCV's findContours often works with (x, y) points.
        # For cv2.convexHull using just points, it expects (N, 2) or (N, 1, 2)
        white_pixels = np.column_stack(np.where(mask_uint8 > 0)[::-1]) # Use [::-1] to get (x, y) from (y, x)

        # Need at least 3 points to form a convex hull
        if len(white_pixels) < 3:
            print("Warning: Not enough pixels (>3) to compute convex hull. Returning original mask.")
            # Convert original binary mask back to float32 0-1 tensor
            final_mask_np = mask_uint8.astype(np.float32) / 255.0
            final_mask_tensor = torch.from_numpy(final_mask_np).unsqueeze(0)
            return (final_mask_tensor,)


        # Compute the convex hull indices
        # hull is a list of indices into the points array
        # OpenCV's cv2.convexHull can return indices or points directly
        # Let's get the points directly
        hull_points = cv2.convexHull(white_pixels) # hull_points shape (N, 1, 2) or (N, 2) depending on OpenCV version/params


        # Create a blank mask to draw the convex hull on
        hull_mask_np = np.zeros_like(mask_uint8)

        # Draw the filled convex hull on the blank mask
        # cv2.fillConvexPoly expects a numpy array of points shape (N, 1, 2) or (N, 2)
        # If hull_points is (N, 1, 2), pass hull_points. If (N, 2), may need reshape or adjustment.
        # Most cv2 functions work fine with (N, 1, 2) from cv2.convexHull default output.
        if hull_points.shape[1] == 1: # Check if it's shape (N, 1, 2)
             cv2.fillConvexPoly(hull_mask_np, hull_points, 255)
        else: # Assume shape (N, 2) and reshape
             cv2.fillConvexPoly(hull_mask_np, hull_points.reshape(-1, 1, 2), 255)


        # Combine the original mask with the hull mask (logical OR)
        # This fills all areas within the convex hull that were not in the original mask
        filled_mask_np = np.maximum(mask_uint8, hull_mask_np)


        # ----------------------------------------------------------
        # Prepare output
        # ----------------------------------------------------------
        # Convert the final mask back to ComfyUI Mask format (B, H, W), float32, 0-1
        final_mask_np = filled_mask_np.astype(np.float32) / 255.0
        final_mask_tensor = torch.from_numpy(final_mask_np).unsqueeze(0)

        return (final_mask_tensor,)