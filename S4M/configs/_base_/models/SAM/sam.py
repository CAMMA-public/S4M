_base_ = "../../default_runtime_iter.py"
custom_imports = dict(
    imports=[
        "mmpretrain.models",
        "S4M.datasets.transforms.prompt",
        "S4M.datasets.transforms.prompt_formatting",
        "S4M.visualization.extreme_area",
        "S4M.models.detectors.SAM",
        "S4M.models.detectors.ExtremeSAM",
        "S4M.models.dense_heads.sam_mask_decoder",
        "S4M.models.utils.sam_layers",
        "S4M.models.task_modules.assigners.SAMassigner",
        "S4M.models.task_modules.prior_generators.prompt_encoder",  # noqa
        "S4M.models.task_modules.prior_generators.label_encoder",  # noqa
        "S4M.hooks.MonkeyPatchHook",
        "S4M.engine.optimizers.layer_decay_optimizer_constructor",  # noqa
        "S4M.evaluation.metrics.instance_iou_metric",
    ],
    allow_failed_imports=False,
)

vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="ExtremeVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
    line_width=12,
    # alpha=0.32,
    alpha=0.12,
)  # noqa

data_preprocessor = dict(
    type="DetDataPreprocessor",
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    bgr_to_rgb=True,
    pad_size_divisor=1024,
)


load_from = (
    "weights/mapped_extreme_sam_vit_b_01ec64.pth"  # load mapped SAM VIT-B weights
)
model = dict(
    type="ExtremeSAM",
    data_preprocessor=data_preprocessor,
    prompt_encoder=dict(
        type="ExtremeSAMPaddingGenerator",
        # type='AngleExtremeSAMPaddingGenerator',
        label_encoder=dict(
            type="LabelEmbedEncoder",
            embed_dims=256,
        ),
    ),
    bbox_head=dict(
        type="SAMHead",
    ),
    backbone=dict(
        type="mmpretrain.ViTSAM",
        arch="base",
        img_size=1024,
        patch_size=16,
        out_channels=256,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        drop_path_rate=0.6,
    ),
    decoder=dict(  # SAMTransformerDecoder
        num_layers=2,
        layer_cfg=dict(  # SAMTransformerLayer
            embedding_dim=256,
            num_heads=8,
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.1),
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            type="SAMassigner",
        )
    ),
)

custom_hooks = [dict(type="MonkeyPatchHook")]
