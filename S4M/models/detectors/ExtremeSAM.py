import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import scale_boxes
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, InstanceList
from mmdet.models.detectors import DetectionTransformer
from mmengine.structures.pixel_data import PixelData
from ..utils.sam_layers import PositionEmbeddingRandom
from S4M.datasets.transforms.prompt_formatting import PackPointDetInputs
from S4M.models.utils.sam_layers import (
    SAMTransformerDecoder,
    SAMTransformerDecoderMultiInstance,
)
from S4M.datasets.transforms.prompt import PromptType
from S4M.models.utils.visualization import dump_masks, dump_fmap
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData
from mmdet.models.utils import unpack_gt_instances
from .SAM import SAM
from mmdet.models.utils import samplelist_boxtype2tensor
from S4M.models.task_modules.prior_generators.canvas_module import Canvas
from S4M.models.task_modules.prior_generators.interaction_simulator import (
    InteractionSimulator,
)
from S4M.visualization.utils import remove_axes, pca, plot_feats, unnormalize_imagenet


@MODELS.register_module()
class ExtremeSAM(SAM):
    """
    SAM predicts object masks from an image and input prompts.

    Arguments:
        backbone (ViTSAM): SAM backbone
        decoder (SAMTransformerDecoder): Transformer Decoder for SAM
        bbox_head (SAMHead): SAM Mask Prediction Head
        prompt_encoder (PromptEncoder): Encodes various types of input prompts.
    """

    def __init__(
        self,
        with_instance_aware_decoder: bool = False,
        num_extra_refinement: int = 0,
        *args,
        **kwargs,
    ):
        self.with_instance_aware_decoder = with_instance_aware_decoder
        self.num_extra_refinement = num_extra_refinement
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        if self.with_instance_aware_decoder:
            self.decoder = SAMTransformerDecoderMultiInstance(**self.decoder)
        else:
            self.decoder = SAMTransformerDecoder(**self.decoder)

        self.canvas = Canvas()

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(ExtremeSAM, self).init_weights()
        self.prompt_encoder.label_encoder.init_weights()

    def forward_decoder(
        self,
        img_feats: Tensor,
        img_pos: Tensor,
        pts_embed: Tensor,
        padded_points: Tensor,
        padded_labels: Tensor,
        attn_mask: Tensor = None,
        prompt_padding_masks: Tensor = None,
        dense_embed: Tensor = None,
        padding_mask: Tensor = None,
        batch_data_samples=None,
        use_canvas_loss="loss",
        **kwargs,
    ) -> Dict:
        """Forward with Transformer decoder.

        Args:

        Returns:
            dict: The dictionary of decoder outputs, which includes
        """
        bs, num_instance, num_query, embed_dim = pts_embed.shape

        src = img_feats[:, None, :, :, :].repeat_interleave(num_instance, dim=1)
        src = src + dense_embed

        # [bs, num_inst, embed_dim, w, h] -> [bs*num_inst, embed_dim, w, h]
        src_shape = src.shape
        src = src.view(-1, *src.shape[2:])

        # [bs, num_inst x num_pts x embed_dim] -> [bs*num_inst x num_pts x embed_dim]
        pts_embed = pts_embed.view(-1, *pts_embed.shape[2:])

        # repeat img_pos for number of instances
        img_pos = img_pos.unsqueeze(0).repeat_interleave(bs * num_instance, dim=0)
        padding_mask = padding_mask.repeat_interleave(num_instance, dim=0).flatten(1)
        prompt_padding_masks = prompt_padding_masks.view(
            bs * num_instance, *prompt_padding_masks.shape[2:]
        )

        if use_canvas_loss is None:
            canvas_out = None
        elif use_canvas_loss == "loss":
            canvas_out = self.canvas.loss(
                img_pos,
                pts_embed,
                prompt_padding_masks,
                padded_labels,
                (bs, num_instance, num_query, *src.shape[1:]),
                batch_data_samples,
            )
        else:
            canvas_out = self.canvas.predict(
                img_pos,
                pts_embed,
                prompt_padding_masks,
                padded_labels,
                (bs, num_instance, num_query, *src.shape[1:]),
                batch_data_samples,
            )

        point_emb, img_emb = self.decoder(
            src,
            img_pos,
            pts_embed,
            padding_mask=None,  # padding_mask,  # I believe its None in SAM repo
            prompt_padding_mask=prompt_padding_masks,  # prompt_padding_masks
        )

        padded_labels = padded_labels.view(bs * num_instance, num_query)
        padded_points = padded_points.view(bs * num_instance, num_query, -1)
        # prompt_padding_masks = prompt_padding_masks.view(bs*num_instance, *prompt_padding_masks.shape[2:])

        # padded label
        head_inputs_dict = dict(
            # shape=src_shape,  # [bs, n_inst, C, h, w]
            shape=src.shape,  # [bs*n_inst, C, h, w]
            point_embedding=point_emb,  # [bs*n_inst, n_prompt, C]
            image_embedding=img_emb,  # [bs*n_inst, img_token, C]
            padded_points=padded_points,  # [bs*n_inst, n_prompt, 2]
            padded_labels=padded_labels,  # [bs*n_inst, n_prompt]
            prompt_padding_masks=prompt_padding_masks,  # [bs*n_inst, n_prompt]
        )

        return head_inputs_dict, canvas_out

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
        use_mask_prompt: bool = False,
        use_canvas_loss=None,
    ) -> Dict:

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples, use_mask_prompt
        )

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)
        decoder_inputs_dict.update(
            dict(batch_data_samples=batch_data_samples, use_canvas_loss=use_canvas_loss)
        )

        decoder_outputs_dict, canvas_out = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        return head_inputs_dict, canvas_out

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        prompter = InteractionSimulator()
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict, canvas_loss = self.forward_transformer(
            img_feats,
            batch_data_samples,
            use_canvas_loss="loss",  # None
        )
        losses, mask_logits = self.bbox_head.loss(
            **head_inputs_dict,
            batch_data_samples=batch_data_samples,
            multimask_output=not self.use_mask_refinement,
        )

        if canvas_loss:
            losses.update(canvas_loss)

        if self.use_mask_refinement:
            for i in range(
                self.num_mask_refinements + self.num_extra_refinement
            ):  # extra pass
                # split mask_logits by img
                instances_per_img = [len(b.gt_instances) for b in batch_data_samples]
                mask_logits = mask_logits.split(instances_per_img)
                new_prompts = prompter.yx.split(instances_per_img)
                new_prompts_types = prompter.point_types.split(instances_per_img)

                # add predicted masks to batch_data_samples
                # add prompts
                for m, np, npt, b in zip(
                    mask_logits, new_prompts, new_prompts_types, batch_data_samples
                ):
                    b.gt_instances.mask_props = m
                    # np tensor of shape shape [n_inst, n_refinement, 2]
                    if i < self.num_mask_refinements:
                        b.gt_instances.interactive_points = np + torch.tensor(
                            [0.5, 0.5], device=new_prompts[0].device
                        )
                        b.gt_instances.interactive_points_types = npt

                # Cascaded Post-refinement
                head_inputs_dict, _ = self.forward_transformer(
                    img_feats,
                    batch_data_samples,
                    use_mask_prompt=True,
                    use_canvas_loss=None,
                )
                losses_ref_stage_i, mask_logits = self.bbox_head.loss(
                    **head_inputs_dict,
                    batch_data_samples=batch_data_samples,
                    # multimask_output=(
                    #     i == self.num_mask_refinements + self.num_extra_refinement - 1
                    # ),
                    multimask_output=False,
                )
                losses_ref_stage_i = {
                    "{}.{}".format(k, i): v for k, v in losses_ref_stage_i.items()
                }
                losses.update(losses_ref_stage_i)

        prompter.reset()
        InteractionSimulator._instance = None
        return losses

    def predict(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> SampleList:
        """Predict results from a batch of inputs and data samples with optional mask refinement"""
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = (
            unpack_gt_instances(batch_data_samples)
        )  # noqa
        if batch_gt_instances[0].bboxes.numel() == 0:
            results = InstanceData()
            results.bboxes = torch.tensor([])
            results.masks = torch.tensor([])
            results.scores = torch.tensor([])
            results.labels = torch.tensor([])
            results.mask_logits = torch.tensor([])

            dummy_mask = torch.full((1, 1, 1), 255, dtype=torch.uint8)
            for b in batch_data_samples:
                b.pred_sem_seg = PixelData(sem_seg=dummy_mask.clone())
                b.gt_sem_seg = PixelData(sem_seg=dummy_mask.clone())

            batch_data_samples = self.add_pred_to_datasample(
                batch_data_samples, [results], [results]
            )
            return batch_data_samples

        # NOTE in train scale factor is 4, as input shape is fixed and downsampled featmap
        # at inference we use the outputed masks in ori shape, so we rescale it to input coordinate
        prompter = InteractionSimulator(mult_factor=1)
        img_feats = self.extract_feat(batch_inputs)
        # mult_factors = [
        #     torch.tensor(meta["scale_factor"][::-1]) for meta in batch_img_metas
        # ]
        mult_factors = [torch.tensor(meta["scale_factor"]) for meta in batch_img_metas]

        head_inputs_dict, canvas_out = self.forward_transformer(
            img_feats,
            batch_data_samples,
            use_canvas_loss="predict",
            # use_canvas_loss=None,
        )
        results_list, image_embedding = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples,
            multimask_output=not self.use_mask_refinement,
            new_prompt=False,  # CARREFULL
        )

        # (Pdb) image_embedding.shape
        # torch.Size([5, 256, 64, 64])
        # batch_inputs[0].shape, image_embedding.shape, batch_img_metas
        # breakpoint()
        # print(batch_inputs.shape, image_embedding.shape)
        results_list_tmp = []

        if self.use_mask_refinement:
            for i in range(
                self.num_mask_refinements + self.num_extra_refinement
            ):  # extra pass
                # split mask_logits by img
                instances_per_img = [len(b.gt_instances) for b in batch_data_samples]

                # add predicted masks to batch_data_samples
                for r, b in zip(results_list, batch_data_samples):
                    b.gt_instances.mask_props = r["mask_logits"]

                # TODO refine the intermediate mask just for plotting, dont propagate
                # Cascaded Post-refinement-1
                ########################################################################################
                if i < self.num_mask_refinements:
                    for zz in range(1):
                        # print(zz, zz == 9)
                        head_inputs_dict, _ = self.forward_transformer(
                            img_feats, batch_data_samples, use_mask_prompt=True
                        )
                        results_list_tmp, _ = self.bbox_head.predict(
                            **head_inputs_dict,
                            rescale=rescale,
                            batch_data_samples=batch_data_samples,
                            multimask_output=(
                                # i == self.num_mask_refinements + self.num_extra_refinement - 1
                                False
                            ),
                            # multimask_output=False,
                            new_prompt=zz == 0,
                        )

                        for r, b in zip(results_list_tmp, batch_data_samples):
                            b.gt_instances.mask_props = r["mask_logits"]

                    # head_inputs_dict, _ = self.forward_transformer(
                    #     img_feats, batch_data_samples, use_mask_prompt=True
                    # )
                    # results_list_tmp, _ = self.bbox_head.predict(
                    #     **head_inputs_dict,
                    #     rescale=rescale,
                    #     batch_data_samples=batch_data_samples,
                    #     multimask_output=(
                    #         # i == self.num_mask_refinements + self.num_extra_refinement - 1
                    #         False
                    #     ),
                    #     new_prompt=False,
                    #     # multimask_output=False,
                    # )

                # UNCOMMENT TO SAVE
                # for r, b in zip(results_list_tmp, batch_data_samples):
                #     if not hasattr(b.gt_instances, "intermediate_masks"):
                #         b.gt_instances.intermediate_masks = r["masks"][:, None]
                #     else:
                #         b.gt_instances.intermediate_masks = torch.cat(
                #             [
                #                 b.gt_instances.intermediate_masks,
                #                 r["masks"][:, None],
                #             ],
                #             dim=1,
                #         )

                # for r, b in zip(results_list, batch_data_samples):
                #     b.gt_instances.mask_props = r["mask_logits"]
                ########################################################################################

                # prompt refinement for num_mask_refinements iter
                if i < self.num_mask_refinements:
                    new_prompts = [
                        x.clone() for x in prompter.yx.split(instances_per_img)
                    ]
                    for idx in range(len(new_prompts)):
                        new_prompts[idx] = (
                            new_prompts[idx] * mult_factors[idx]
                        ) + torch.tensor([0.5, 0.5], device=new_prompts[idx].device)

                    # add new prompts
                    new_prompts_types = prompter.point_types.split(instances_per_img)
                    for np, npt, b in zip(
                        new_prompts, new_prompts_types, batch_data_samples
                    ):
                        # np tensor of shape shape [n_inst, n_refinement, 2]
                        b.gt_instances.interactive_points = np
                        b.gt_instances.interactive_points_types = npt

                # Cascaded Post-refinement-1
                head_inputs_dict, _ = self.forward_transformer(
                    img_feats, batch_data_samples, use_mask_prompt=True
                )
                results_list, image_embedding = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples,
                    multimask_output=(
                        # i == self.num_mask_refinements + self.num_extra_refinement - 1
                        False
                    ),
                    new_prompt=False,
                    # post_pro_smooth=True,
                    # multimask_output=False,
                )

        # plot_feats(
        #     unnormalize_imagenet(batch_inputs[0]),
        #     img_feats[-1],
        #     image_embedding,
        #     f"./outputs/vis/{batch_img_metas[0]['img_id']}.png",
        # )

        # convert mask to semantic segmentation mask and store for metric computation
        # for b, r in zip(batch_data_samples, results_list):
        #     # convert pred
        #     inst_masks = r['masks']
        #     scores = r['scores']
        #     labels = r['labels'] + 1
        #     fg = (inst_masks > 0.5).any(0).float()
        #     pixel_to_label = (inst_masks * scores.view(-1, 1, 1)).argmax(0)
        #     sem_mask = labels[pixel_to_label] * fg
        #     # b.pred_sem_seg = PixelData(sem_seg=sem_mask.unsqueeze(0))

        #     # convert gt
        #     inst_masks = b.gt_instances.masks.to_tensor(dtype=inst_masks.dtype,
        #             device=inst_masks.device)
        #     fg = (inst_masks == 1).any(0).float()
        #     pixel_to_label = (inst_masks * scores.view(-1, 1, 1)).argmax(0)
        #     sem_mask = labels[pixel_to_label] * fg
        #     b.gt_sem_seg = PixelData(sem_seg=sem_mask.unsqueeze(0))
        # breakpoint()

        # convert mask to semantic segmentation mask and store for metric computation
        for b, r in zip(batch_data_samples, results_list):
            # convert pred
            inst_masks = r["masks"]
            scores = r["scores"]
            labels = r["labels"] + 1
            fg = (inst_masks > 0.5).any(0).float()
            pixel_to_label = (inst_masks * scores.view(-1, 1, 1)).argmax(0)
            sem_mask = labels[pixel_to_label] * fg
            b.pred_sem_seg = PixelData(sem_seg=sem_mask.unsqueeze(0))

            # convert gt
            inst_masks = b.gt_instances.masks.to_tensor(
                dtype=inst_masks.dtype, device=inst_masks.device
            )
            fg = (inst_masks == 1).any(0).float()
            pixel_to_label = (inst_masks * scores.view(-1, 1, 1)).argmax(0)
            sem_mask = labels[pixel_to_label] * fg
            b.gt_sem_seg = PixelData(sem_seg=sem_mask.unsqueeze(0))

        # for b, r in zip(batch_data_samples, results_list):
        #     # === PRED ===
        #     inst_masks = r['masks']  # (N, H, W)
        #     N = inst_masks.shape[0]
        #     fg = (inst_masks > 0.5).any(0).float()
        #     pixel_to_inst = inst_masks.float().argmax(0)
        #     sem_mask = torch.zeros_like(pixel_to_inst)
        #     for i in range(N):
        #         sem_mask[pixel_to_inst == i] = i + 1  # instance ID starts from 1
        #     sem_mask = sem_mask * fg
        #     b.pred_sem_seg = PixelData(sem_seg=sem_mask.unsqueeze(0))

        #     # === GT ===
        #     inst_masks = b.gt_instances.masks.to_tensor(dtype=inst_masks.dtype, device=inst_masks.device)
        #     N_gt = inst_masks.shape[0]
        #     fg = (inst_masks == 1).any(0).float()
        #     pixel_to_inst = inst_masks.float().argmax(0)
        #     sem_mask = torch.zeros_like(pixel_to_inst)
        #     for i in range(N_gt):
        #         sem_mask[pixel_to_inst == i] = i + 1  # instance ID starts from 1
        #     sem_mask = sem_mask * fg
        #     b.gt_sem_seg = PixelData(sem_seg=sem_mask.unsqueeze(0))

        prompter.reset()
        InteractionSimulator._instance = None
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list, canvas_out
        )
        # print(batch_data_samples)
        # print("\n====\n")

        return batch_data_samples

    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        results_list: InstanceList,
        canvas_list: InstanceList,
    ) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        for i, (data_sample, pred_instances) in enumerate(
            zip(data_samples, results_list)
        ):
            data_sample.pred_instances = pred_instances
            if canvas_list is not None:
                data_sample.pred_instances_canvas = canvas_list[i]

        samplelist_boxtype2tensor(data_samples)
        return data_samples
