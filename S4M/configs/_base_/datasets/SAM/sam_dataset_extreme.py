# dataset settings
dataset_type = "CocoDataset"
file_client_args = dict(backend="disk")
backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", file_client_args=file_client_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(
        type="GetEdgeMask",
    ),
    dict(type="GetAnatomicalPolesFixedAxis", w_main=1.0, w_ortho=0.0, test=False),
    dict(type="PackDummyDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", file_client_args=file_client_args),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="GetEdgeMask",
    ),
    dict(
        type="GetSkeletonMask",
    ),
    dict(
        type="GetAnatomicalPolesFixedAxis",
        w_main=1.0,
        w_ortho=0.0,
        test=True,
        #  pole_erosion=15,
    ),
    dict(
        type="PackDummyDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "anatomical_pole_area",
            "pca_centers",
            "pca_axes",
            "skeleton_masks",
            "candidates",
        ),
    ),
]

dummy_metainfo = {"classes": ("object",)}

train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        metainfo=dummy_metainfo,
        data_prefix=dict(img="images/train/"),
        ann_file="annotations/train/annotation_coco.json",
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        # indices=200,
        metainfo=dummy_metainfo,
        pipeline=test_pipeline,
        data_prefix=dict(img="images/val/"),
        ann_file="annotations/val/annotation_coco.json",
        # filter_cfg=dict(filter_empty_gt=True),
        test_mode=True,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = [
    dict(type="CocoMetric", metric=["bbox", "segm"], format_only=False, classwise=True),
    dict(
        # type="InstanceIoUMetric",
        type="InstanceIoUConcavityMetric",
        metric=["bbox", "segm"],
        format_only=False,
        classwise=True,
    ),
    dict(
        # type='InstanceSegMetric',
        type="SemSegMetric",
        iou_metrics=["mIoU", "mDice", "mFscore"],
        collect_device="gpu",
    ),
]

test_evaluator = val_evaluator
