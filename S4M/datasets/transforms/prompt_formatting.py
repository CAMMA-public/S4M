from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs, PackTrackInputs


@TRANSFORMS.register_module()
class PackDummyDetInputs(PackDetInputs):

    mapping_table = {
        "gt_bboxes": "bboxes",
        "gt_bboxes_labels": "labels",
        "gt_masks": "masks",
        "anatomical_pole_pools": "anatomical_pole_pools",
        # 'anatomical_pole_area': 'anatomical_pole_area'
    }


@TRANSFORMS.register_module()
class PackMixedPromptInputs(PackDetInputs):

    mapping_table = {
        "gt_bboxes": "bboxes",
        "gt_bboxes_labels": "labels",
        "gt_masks": "masks",
        "anatomical_pole_pools": "anatomical_pole_pools",
        "prompt_types": "prompt_types",
    }


@TRANSFORMS.register_module()
class PackSkeletonDetInputs(PackDetInputs):

    mapping_table = {
        "gt_bboxes": "bboxes",
        "gt_bboxes_labels": "labels",
        "gt_masks": "masks",
        "anatomical_pole_pools": "anatomical_pole_pools",
        "skeleton_sampled_points": "skeleton_sampled_points",
    }


@TRANSFORMS.register_module()
class PackPointDetInputs(PackDetInputs):

    mapping_table = {
        "gt_bboxes": "bboxes",
        "gt_bboxes_labels": "labels",
        "gt_masks": "masks",
        "points": "points",
        "anatomical_pole_pools": "anatomical_pole_pools",
        # 'boxes': 'boxes',
        # 'prompt_types': 'prompt_types'
    }


@TRANSFORMS.register_module()
class PackDummyTrackInputs(PackTrackInputs):

    mapping_table = {
        "gt_bboxes": "bboxes",
        "gt_bboxes_labels": "labels",
        "gt_masks": "masks",
        "anatomical_pole_pools": "anatomical_pole_pools",
    }
