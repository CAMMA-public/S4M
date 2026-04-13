# AMP optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    # loss goes to NaN with AMP
    # type='AmpOptimWrapper',
    # loss_scale='dynamic',
    # dtype="float16",
    optimizer=dict(type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=1600, val_interval=200)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=1600,
        by_epoch=False,
        milestones=[1000, 1400],
        gamma=0.1,
    ),
]
