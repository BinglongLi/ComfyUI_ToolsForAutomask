from .DirectionalMaskExpansionCustomNode import DirectionalMaskExpansionCustomNode
from .RemoveSmallRegionsMaskNode import RemoveSmallRegionsMaskNode
from .operations import *
from .PruneThinBranchesMaskNode import PruneThinBranchesMaskNode
from .MaskFillGapsConvexHullNode import MaskFillGapsConvexHullNode


NODE_CLASS_MAPPINGS = {
    "Directional Mask Expansion": DirectionalMaskExpansionCustomNode,
    "Remove Small Regions Mask": RemoveSmallRegionsMaskNode,
    "Precise Subtract Mask": PreciseSubtractMaskNode,
    "Precise Add Mask": PreciseAddMaskNode,
    "Closing Mask": ClosingMaskNode,
    "Opening Mask": OpeningMaskNode,
    "Conditional Mask Selector": ConditionalMaskSelector,
    "Prune Thin Branches Mask": PruneThinBranchesMaskNode,
    "Mask Fill Gaps Convex Hull": MaskFillGapsConvexHullNode,
}

__all__ = ['NODE_CLASS_MAPPINGS']