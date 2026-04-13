from typing import Dict, List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule, Sequential, xavier_init
from mmengine.structures import InstanceData
from torch import Tensor
from scipy import ndimage as ndi

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.mask import mask2bbox
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import (
    ConfigType,
    InstanceList,
    OptInstanceList,
    OptMultiConfig,
    reduce_mean,
)
from mmdet.models.utils import multi_apply
from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.models.layers.transformer import MLP
from S4M.models.task_modules.prior_generators.prompt_encoder import (
    ExtremeEmbeddingIndex,
    EmbeddingIndex,
)
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from S4M.models.utils.sam_layers import SAMTransformerDecoder
from S4M.models.dense_heads.sam_mask_decoder import LayerNorm2d
from mmengine.model import bias_init_with_prob, constant_init
import numpy as np
import cv2


@MODELS.register_module()
class CanvasHead(BaseModule):
    def __init__(
        self,
        *,
        transformer_dim: int = 256,
        activation: Type[nn.Module] = nn.GELU,
        init_cfg: OptMultiConfig = None,
        **kwargs
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.transformer_dim = transformer_dim
        self.activation = activation
        self._init_layers()

    def _init_layers(self) -> None:
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(
                    self.transformer_dim,
                    self.transformer_dim,
                    self.transformer_dim // 8,
                    3,
                )
            ]
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                self.transformer_dim, self.transformer_dim // 4, kernel_size=2, stride=2
            ),  # noqa
            LayerNorm2d(self.transformer_dim // 4),
            self.activation(),
            nn.ConvTranspose2d(
                self.transformer_dim // 4,
                self.transformer_dim // 8,
                kernel_size=2,
                stride=2,
            ),  # noqa
            self.activation(),
        )

    def forward(self, act_image_embedding: Tensor, mask_tokens_out: Tensor) -> Tensor:
        act_image_embedding = self.output_upscaling(act_image_embedding)
        hyper_in = self.output_hypernetworks_mlps[0](mask_tokens_out[:, None, :])
        return act_image_embedding, hyper_in


@MODELS.register_module()
class Canvas(BaseModule):
    """
    Auxiliary module to predict segmentation masks from prompt domain alone
    """

    def __init__(
        self,
        *,
        decoder: OptConfigType = dict(
            skip_first_layer_pe=False,
            num_layers=2,
            layer_cfg=dict(  # SAMTransformerLayer
                embedding_dim=256,
                num_heads=8,
                ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.1),
            ),
            # init_cfg=dict(
            #     type="Pretrained",
            #     prefix="decoder.",
            #     checkpoint="weights/mapped_sam_vit_b_01ec64.pth",
            # ),
        ),
        transformer_dim: int = 256,
        activation: Type[nn.Module] = nn.GELU,
        loss_mask_focal: ConfigType = dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=20.0,
        ),
        loss_mask_dice: ConfigType = dict(
            type="DiceLoss",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            eps=1.0,
            loss_weight=1.0,
        ),
        train_cfg: ConfigType = dict(
            assigner=dict(
                type="SAMassigner",
            )
        ),
        canvas_head: ConfigType = dict(
            type="CanvasHead",
            # init_cfg=dict(
            #     type="Pretrained",
            #     prefix="bbox_head.",
            #     checkpoint="weights/mapped_sam_vit_b_01ec64.pth",
            # ),
        ),
        init_cfg: OptMultiConfig = None,
        **kwargs
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.transformer_dim = transformer_dim
        self.activation = activation
        self.decoder = decoder

        self.train_cfg = train_cfg
        self.loss_mask_focal = MODELS.build(loss_mask_focal)
        self.loss_mask_dice = MODELS.build(loss_mask_dice)
        self.canvas_head = MODELS.build(canvas_head)

        self._init_layers()

    def _init_layers(self) -> None:
        # self.decoder = SAMSyncTransformerDecoder(**self.decoder,)
        self.decoder = SAMTransformerDecoder(**self.decoder)
        # self.decoder = SAMTransformerDecoder(**self.decoder)
        # self.canvas_img_embedding = nn.Embedding(256, 64 * 64)
        # self.canvas_img_embedding = nn.Embedding(64 * 64, 256)
        self.canvas_img_embedding = nn.Embedding(1, 256)
        # TODO learn 1 token embedding, and dupplicate it for every patch of the passed img
        # nn.init.xavier_uniform_(self.canvas_img_embedding.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.canvas_img_embedding.weight)

    def forward(
        self,
        img_pos: Tensor,
        pts_embed: Tensor,
        prompt_padding_masks: Tensor,
        padded_labels: Tensor,
        shape: Tuple,
    ) -> Tensor:
        """
        Forward pass of the Canvas.

        Args:
            prompt_padding_masks  # [bs, num_instance, num_points] 1 indicate is padding
            pts_embed  # [bs*num_inst x num_pts x embed_dim] embedding, including pos

        Returns:
            Tensor: The embedding of the labels.
        """
        b, num_instance, num_query, c, h, w = shape

        canvas_img_embedding = self.canvas_img_embedding.weight  # (1, 256)
        canvas_img_embedding = canvas_img_embedding.expand(64 * 64, -1)  # (4096, 256)
        canvas_img_embedding = canvas_img_embedding.view(64, 64, 256)  # (64, 64, 256)
        canvas_img_embedding = (
            canvas_img_embedding.unsqueeze(0)
            .expand(b * num_instance, -1, -1, -1)
            .permute(0, 3, 1, 2)
        )  # (bs, 256, 64, 64)

        prompt_padding_masks = prompt_padding_masks.view(b * num_instance, num_query)
        padded_labels = padded_labels.view(b * num_instance, num_query)

        point_embedding, image_embedding = self.decoder(
            canvas_img_embedding,
            # torch.zeros_like(img_pos),
            # torch.ones_like(img_pos),
            # img_pos,
            img_pos,
            # torch.ones_like(img_pos),
            # torch.zeros_like(img_pos),
            # img_pos,
            pts_embed,
            # pts_embed[(padded_labels == ExtremeEmbeddingIndex.TOP.value)
            #         | (padded_labels == ExtremeEmbeddingIndex.BOTTOM.value)
            #         | (padded_labels == ExtremeEmbeddingIndex.LEFT.value)
            #         | (padded_labels == ExtremeEmbeddingIndex.RIGHT.value)
            #         | (padded_labels == ExtremeEmbeddingIndex.CANVAS.value)],
            padding_mask=None,  # padding_mask,  # I believe its None in SAM repo
            # prompt_padding_mask=None,  # prompt_padding_masks
            prompt_padding_mask=prompt_padding_masks,  # prompt_padding_masks
        )

        # then head
        active_prompts = ~(prompt_padding_masks.bool())  # per instance & prompt padding
        active_inputs = ~torch.all(~active_prompts, 1)  # per instance padding
        act_point_embedding = point_embedding[active_inputs]
        act_image_embedding = image_embedding[active_inputs]
        act_padded_labels = padded_labels[active_inputs]

        act_b = active_inputs.sum()

        # find CANVAS token
        mask_tokens_out = act_point_embedding[
            act_padded_labels
            == ExtremeEmbeddingIndex.CANVAS.value
            # act_padded_labels
            # == EmbeddingIndex.MASK_OUT.value
        ]
        # mask_tokens_out = act_point_embedding[act_padded_labels == EmbeddingIndex.MASK_OUT.value]
        act_image_embedding = act_image_embedding.transpose(1, 2).view(act_b, c, h, w)

        act_image_embedding, hyper_in = self.canvas_head(
            act_image_embedding, mask_tokens_out
        )

        b, c, h, w = act_image_embedding.shape
        masks = (hyper_in @ act_image_embedding.view(act_b, c, h * w)).view(
            act_b, -1, h, w
        )  # noqa
        return masks

    def predict(
        self,
        img_pos: Tensor,
        pts_embed: Tensor,
        prompt_padding_masks: Tensor,
        padded_labels: Tensor,
        shape: Tuple,
        batch_data_samples: SampleList,
        rescale: bool = True,
    ):
        mask_logits = self(
            img_pos, pts_embed, prompt_padding_masks, padded_labels, shape
        )[:, slice(0, 1), :, :]

        # get masks from logits
        masks = mask_logits > 0
        results = []
        # split masks, scores, and mask_logits by img
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        instances_per_img = [len(b.gt_instances) for b in batch_data_samples]
        # breakpoint()
        masks = list(masks.split(instances_per_img))
        mask_logits = mask_logits.split(instances_per_img)

        def largest_cc_fill_smooth(
            mask: torch.Tensor,
            thr=0.5,
            do_close=True,
            smooth: str | None = "dt",  # "dt", "morph", or None
            sigma: float = 1.5,  # for "dt"
            morph_radius: int = 0,  # for "morph" (0=auto small)
        ) -> torch.Tensor:
            """
            Keep largest connected component, fill holes, then (optional) smooth borders.
            Input/Output shape preserved: (H,W), (C,H,W), or (B,C,H,W).
            """
            device, dtype = mask.device, mask.dtype
            *lead, H, W = mask.shape
            flat = mask.reshape(-1, H, W)

            st8 = np.ones((3, 3), dtype=bool)

            def _proc(x: torch.Tensor) -> torch.Tensor:
                m = (x > thr).detach().cpu().numpy().astype(np.uint8)

                # largest CC (8-connexity)
                lbl, num = ndi.label(m, structure=st8)
                if num > 0:
                    sizes = ndi.sum(m, lbl, index=np.arange(1, num + 1))
                    largest = lbl == (sizes.argmax() + 1)
                else:
                    largest = m.astype(bool)

                # optional small gap closing before fill
                if do_close:
                    largest = ndi.binary_closing(largest, structure=st8, iterations=1)

                filled = ndi.binary_fill_holes(largest, structure=st8)

                # --- smoothing ---
                if smooth == "dt":
                    # signed distance -> gaussian -> threshold
                    inside = filled.astype(np.uint8)
                    sdf = ndi.distance_transform_edt(
                        inside
                    ) - ndi.distance_transform_edt(1 - inside)
                    sdf_s = ndi.gaussian_filter(sdf, sigma=max(0.01, float(sigma)))
                    smoothed = sdf_s > 0
                    out_np = smoothed.astype(np.float32)
                elif smooth == "morph":
                    r = morph_radius if morph_radius > 0 else 1
                    se = ndi.iterate_structure(ndi.generate_binary_structure(2, 1), r)
                    closed = ndi.binary_closing(filled, structure=se)
                    opened = ndi.binary_opening(closed, structure=se)
                    out_np = opened.astype(np.float32)
                else:
                    out_np = filled.astype(np.float32)

                return torch.from_numpy(out_np)

            out = torch.stack([_proc(x) for x in flat], dim=0)
            return out.reshape(*lead, H, W).to(device).type(dtype)

        for ind, b in enumerate(batch_img_metas):
            if rescale:
                pad_shape = b["pad_shape"]
                img_shape = b["img_shape"]
                ori_shape = b["ori_shape"]

                # breakpoint()
                # first resize mask to pad shape
                masks[ind] = F.interpolate(
                    masks[ind].float(), pad_shape, mode="bilinear", align_corners=False
                )

                # now crop to img shape
                masks[ind] = masks[ind][..., : img_shape[0], : img_shape[1]]

                # finally resize to ori_shape
                masks[ind] = F.interpolate(
                    masks[ind], ori_shape, mode="bilinear", align_corners=False
                )
                masks[ind] = largest_cc_fill_smooth(
                    masks[ind], thr=0.5, smooth="dt", sigma=2
                )

            # cast masks and extract bboxes
            masks[ind] = masks[ind].squeeze(1).bool()
            bboxes = mask2bbox(masks[ind])

            result = InstanceData()
            result.bboxes = bboxes
            result.labels = batch_data_samples[ind].gt_instances.labels
            result.masks = masks[ind]
            result.mask_logits = mask_logits[ind]
            # results.append(InstanceData(masks=masks[ind], bboxes=bboxes, scores=scores[ind],
            #     # point_pred=b.gt_instances[0].points,
            #     labels=b.gt_instances.labels, mask_logits=mask_logits[ind]))
            results.append(result)

        return results

    def loss(
        self,
        img_pos: Tensor,
        pts_embed: Tensor,
        prompt_padding_masks: Tensor,
        padded_labels: Tensor,
        shape: Tuple,
        batch_data_samples: SampleList,
    ) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            hidden_states (Tensor): Feature from the transformer decoder, has
                shape (num_decoder_layers, bs, num_queries, cls_out_channels)
                or (num_decoder_layers, num_queries, bs, cls_out_channels).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        mask_logits = self(
            img_pos, pts_embed, prompt_padding_masks, padded_labels, shape
        )[:, 0, :, :]
        # Example tensor `masks` (assuming it's already defined)
        # binary_masks = (mask_logits > 0).to(torch.uint8) * 255  # Convert to 0 (black) and 255 (white)

        # # Save each mask in the batch
        # for i, binary_mask in enumerate(binary_masks):
        #     breakpoint()
        #     # TODO save gt masks

        #     img = Image.fromarray(binary_mask.squeeze().cpu().numpy(), mode="L")  # Convert to grayscale
        #     img.save(f"./plots_recon_canvas/{data_sample.metainfo['img_path'].split('/')[-1].split('.')[0]}___{i}.png")

        loss_inputs = (mask_logits, batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_by_feat(
        self,
        mask_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        The IoU prediction head is trained with mean-square-error loss
        between the IoU prediction and the predicted mask’s IoU with the ground truth mask

        Returns:
            Tuple[Tensor]: A tuple including `loss_score`, `loss_iou`.
        """

        # get targets
        selected_pred_masks, resized_gt_masks = self._get_targets_single(
            mask_preds, batch_gt_instances
        )

        # TODO based on batch gt instance, ignore loss for point or box (we train canvas only with extreme points)

        # dict for losses
        losses = {}

        # compute dice loss
        loss_dice = self.loss_mask_dice(selected_pred_masks, resized_gt_masks)
        losses.update({"canvas.loss_dice": loss_dice})

        # compute focal loss
        # loss_focal = self.loss_mask_focal(selected_pred_masks.flatten(1), resized_gt_masks.flatten(1).long())
        loss_focal = self.loss_mask_focal(selected_pred_masks, resized_gt_masks)
        losses.update({"canvas.loss_focal": loss_focal})

        return losses

    def _get_targets_single(
        self, mask_preds: Tensor, gt_instances: InstanceData
    ) -> tuple:
        """rescale mask, compute IOU as target for iou token,
        only best mask is returned
        """
        # get gt masks
        gt_masks = [
            b.masks.to_tensor(device=mask_preds.device, dtype=mask_preds.dtype)
            for b in gt_instances
        ]

        # pad gt masks
        # NOTE could be done in preprocessor, but need to be done only on training data
        target_size = max(gt_instances[0].masks.width, gt_instances[0].masks.height)
        padded_gt_masks = torch.cat(
            [
                F.pad(m, (0, target_size - m.shape[2], 0, target_size - m.shape[1]))
                for m in gt_masks
            ]
        )

        # resize gt masks
        resized_gt_masks = (
            F.interpolate(
                padded_gt_masks.unsqueeze(0),
                size=mask_preds.shape[-2:],
                mode="bilinear",
            )
            .squeeze(0)
            .round()
        )
        convex_gt_masks = get_convex_hull_masks(resized_gt_masks)
        return mask_preds, convex_gt_masks  # resized_gt_masks
        # return mask_preds, resized_gt_masks


# TODO do convex hull on high res img, not lower res
# Assume resized_gt_masks is a tensor of shape (N, H, W), where N is the number of masks
def get_convex_hull_masks(resized_gt_masks):
    convex_masks = torch.zeros_like(resized_gt_masks)  # Placeholder for new masks

    for i in range(resized_gt_masks.shape[0]):  # Iterate over batch
        mask_np = resized_gt_masks[i].cpu().numpy().astype(np.uint8)  # Convert to NumPy

        # Find contours
        contours, _ = cv2.findContours(
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:  # Check if contours were found
            # Compute convex hull
            hull = [cv2.convexHull(cnt) for cnt in contours]

            # Create an empty mask and draw the convex hull
            convex_mask = np.zeros_like(mask_np)
            cv2.drawContours(convex_mask, hull, -1, 1, thickness=-1)

            # Convert back to tensor
            convex_masks[i] = torch.tensor(convex_mask, dtype=torch.float32)

    return convex_masks
