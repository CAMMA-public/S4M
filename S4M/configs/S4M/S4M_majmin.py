_base_ = ['../_base_/datasets/SAM/sam_dataset_majmin.py',
          '../_base_/models/SAM/sam_mask_refinement.py',
          '../_base_/schedules/default.py']

optim_wrapper = dict(
    constructor='SAMLearningRateDecayOptimizerConstructor',
    paramwise_cfg=dict(
        # decay_rate=0.8,
        decay_rate=0.6,
        decay_type='layer_wise',
        num_layers=12,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
        }
    )
)
