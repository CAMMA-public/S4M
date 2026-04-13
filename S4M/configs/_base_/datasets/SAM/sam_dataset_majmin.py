_base_ = [
    "./sam_dataset_extreme.py",
]

file_client_args = dict(backend="disk")

train_pipeline = [
    dict(type="LoadImageFromFile", file_client_args=file_client_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(
        type="GetEdgeMask",
    ),
    dict(
        type="GetSkeletonMask",
    ),
    dict(
        type="GetAnatomicalPoles",
        #  top_x=0.20,
        w_main=0.6,
        w_ortho=0.4,
        #  w_main=1., w_ortho=0.,
        test=False,
    ),
    dict(
        type="PackDummyDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "pca_centers",
            "pca_axes",
        ),
    ),
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
        type="GetAnatomicalPoles",
        w_main=0.6,
        w_ortho=0.4,
        #  pole_erosion=15,
        #  w_main=1., w_ortho=0.,
        test=True,
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
        ),
    ),
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
