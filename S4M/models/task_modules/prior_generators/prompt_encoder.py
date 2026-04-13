from typing import Dict, List, Tuple, Union, Optional, Any, Type

import torch
from torch import nn
import numpy as np
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType
from S4M.datasets.transforms.prompt_formatting import PackPointDetInputs
from mmdet.models.layers.transformer.utils import MLP
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, InstanceList
from mmdet.models.utils import unpack_gt_instances
from S4M.models.utils.sam_layers import PositionEmbeddingRandom, LayerNorm2d
from enum import Enum, auto
from S4M.models.task_modules.prior_generators.instance_encoder import (
    InstanceEmbedEncoder,
    InstanceEmbedEncoding,
)
from S4M.datasets.transforms.prompt import PromptType

prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size


class EmbeddingIndex(Enum):
    NON_INIT_MASK_EMBED = 0
    NEG = auto()
    POS = auto()
    BOX_CORNER_A = auto()
    BOX_CORNER_B = auto()
    NOT_A_POINT = auto()
    MASK_OUT = auto()  # embedding for non_ambiguous mask
    MASK_OUT_1 = (
        auto()
    )  # need a separate embedding for each of the 3 output mask granularities
    MASK_OUT_2 = auto()
    MASK_OUT_3 = auto()
    IOU_OUT = auto()


class ExtremeEmbeddingIndex(Enum):
    # Start from the next value after EmbeddingIndex.IOU_OUT
    # TOP = EmbeddingIndex.IOU_OUT.value + 1
    # BOTTOM = TOP + 1
    # LEFT = BOTTOM + 1
    # RIGHT = LEFT + 1
    # CANVAS = RIGHT + 1
    # MAJ = CANVAS + 1
    # MIN = MAJ + 1
    TOP = EmbeddingIndex.IOU_OUT.value + 1
    BOTTOM = TOP + 1
    LEFT = BOTTOM + 1
    RIGHT = LEFT + 1
    CANVAS = RIGHT + 1
    INSTANCE = CANVAS + 1


@MODELS.register_module()
class SAMPaddingGenerator(BaseModule):
    """
    A generator for padding tensors.
    """

    def __init__(
        self,
        embed_dim: int = prompt_embed_dim,
        mask_in_chans: int = 16,
        activation: Type[nn.Module] = nn.GELU,
        image_embedding_size: Tuple[int, int] = (
            image_embedding_size,
            image_embedding_size,
        ),
        input_image_size: Tuple[int, int] = (image_size, image_size),
        label_encoder: OptConfigType = None,
        n_output_tokens: int = 5,  # 3 + 1 (non_ambiguiti mask token) + 1
        use_mask_refinement: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        """
        Initialize the PointPaddingGenerator.

        Args:
            mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
            max_num_instances (int): Maximum number of instances. Default: 100.
            max_num_points (int): Maximum number of points. Default: 1.
            embed_dim (int): The prompts' embedding dimension
            NOTE i guess this is of fixed size? same for all batch?
            image_embedding_size (tuple(int, int)): The spatial size of the
                image embedding, as (H, W).
            input_image_size (int): The padded size of the image as input
                to the image encoder, as (H, W).
            init_cfg (OptMultiConfig): Configuration ford initialization. Default: None.
        """
        super().__init__(init_cfg=init_cfg)
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.n_output_tokens = n_output_tokens  # (multimask + iou) tokens
        self.mask_in_chans = mask_in_chans
        self.activation = activation

        label_encoder["num_classes"] = len(
            EmbeddingIndex
        )  # non_init_mask_embed, pos, neg, box_corner_a, box_corner_b, not_a_point, 4 mask_outs, iou_out
        label_encoder["embed_dims"] = self.embed_dim
        self.label_encoder = MODELS.build(label_encoder)
        self.use_mask_refinement = use_mask_refinement

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize the embedding layer."""
        if self.use_mask_refinement:
            self.mask_downscaling = nn.Sequential(
                nn.Conv2d(1, self.mask_in_chans // 4, kernel_size=2, stride=2),
                LayerNorm2d(self.mask_in_chans // 4),
                self.activation(),
                nn.Conv2d(
                    self.mask_in_chans // 4, self.mask_in_chans, kernel_size=2, stride=2
                ),
                LayerNorm2d(self.mask_in_chans),
                self.activation(),
                nn.Conv2d(self.mask_in_chans, self.embed_dim, kernel_size=1),
            )

    def _init_instance_attention_mask(self, max_num_prompts: int) -> torch.Tensor:
        # Each instance has a mask for points + output tokens
        size = max_num_prompts
        return torch.zeros(size, size)

    def create_global_attention_mask(
        self, max_num_instances: int, max_num_prompts: int
    ) -> torch.Tensor:
        # Initialize a large attention matrix to cover all instances
        total_size = max_num_prompts * max_num_instances
        global_attention_mask = torch.ones(total_size, total_size)

        # Compute the base instance mask once
        instance_mask = self._init_instance_attention_mask(max_num_prompts)

        # Apply the padding to the base mask and place updated masks into the global mask
        for i in range(max_num_instances):
            start_index = i * max_num_prompts
            end_index = start_index + max_num_prompts
            global_attention_mask[start_index:end_index, start_index:end_index] = (
                instance_mask
            )

        return global_attention_mask

    def process_prompt(
        self,
        padding_masks: torch.Tensor,
        attn_mask: torch.Tensor,
        padded_points: torch.Tensor,
        padded_contents: torch.Tensor,
        dense_embeddings: torch.Tensor,
        with_instance_idx: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process point data by reshaping and applying content and XY encoders.

        Parameters:
        padding_masks (torch.Tensor): The tensor for padding masks.
        attn_mask (torch.Tensor): The attention mask tensor.
        padded_points (torch.Tensor): The tensor containing padded points.
        padded_contents (torch.Tensor): The tensor containing padded contents.
        with_instance_idx: return rensor with instance idx or not

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing the reshaped padding_masks, attn_mask, padded_points,
            and the result of adding pos_padded_points with embed_padded_contents.
        """
        # Extract dimensions from padded points
        bs, num_instance, num_points, xy_dims = padded_points.shape

        # print(padded_points[0, 0])
        # print(padded_contents[0, 0])

        # Reshape the tensors to flatten the points and contents dimensions
        padding_masks = padding_masks.view(bs, num_instance * num_points)
        padded_points = padded_points.view(bs, num_instance * num_points, xy_dims)
        padded_labels = padded_contents.view(bs, num_instance * num_points)

        # for padded and output tokens: no pos embed + embedding not a point
        pts_embed = self.label_encoder(padded_labels)
        pos_embed = self.pe_layer.forward_with_coords(
            padded_points, self.input_image_size
        )

        is_mask_token = (
            (padded_labels == EmbeddingIndex.MASK_OUT.value)
            + (padded_labels == EmbeddingIndex.MASK_OUT_1.value)
            + (padded_labels == EmbeddingIndex.MASK_OUT_2.value)
            + (padded_labels == EmbeddingIndex.MASK_OUT_3.value)
        ) > 0
        is_iou_token = padded_labels == EmbeddingIndex.IOU_OUT.value
        is_not_a_point_token = padded_labels == EmbeddingIndex.NOT_A_POINT.value
        is_canvas_token = padded_labels == ExtremeEmbeddingIndex.CANVAS.value
        # is_instance_token = padded_labels == ExtremeEmbeddingIndex.INSTANCE.value

        pos_embed[
            is_mask_token
            | is_iou_token
            | is_not_a_point_token
            | is_canvas_token
            # | is_instance_token
        ] = 0.0

        pts_embed = pts_embed + pos_embed

        # for SAM we need to have instances indexing
        if with_instance_idx:
            padding_masks = padding_masks.view(bs, num_instance, num_points)
            padded_points = padded_points.view(
                bs, num_instance, num_points, xy_dims
            )  # noqa
            pts_embed = pts_embed.view(bs, num_instance, num_points, -1)
            padded_labels = padded_labels.view(bs, num_instance, num_points)

        # padded_labels indicate type of prompt (point, mask, iou..)
        return (
            padding_masks,
            attn_mask,
            padded_points,
            padded_labels,
            pts_embed,
            dense_embeddings,
        )

    def process_prompt_with_instance(
        self,
        padding_masks: torch.Tensor,
        attn_mask: torch.Tensor,
        padded_points: torch.Tensor,
        padded_contents: torch.Tensor,
        dense_embeddings: torch.Tensor,
        with_instance_idx: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process point data by reshaping and applying content and XY encoders.

        Parameters:
        padding_masks (torch.Tensor): The tensor for padding masks.
        attn_mask (torch.Tensor): The attention mask tensor.
        padded_points (torch.Tensor): The tensor containing padded points.
        padded_contents (torch.Tensor): The tensor containing padded contents.
        with_instance_idx: return rensor with instance idx or not

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing the reshaped padding_masks, attn_mask, padded_points,
            and the result of adding pos_padded_points with embed_padded_contents.
        """
        # Extract dimensions from padded points
        bs, num_instance, num_points, xy_dims = padded_points.shape

        # Reshape the tensors to flatten the points and contents dimensions
        padding_masks = padding_masks.view(bs, num_instance * num_points)
        padded_points = padded_points.view(bs, num_instance * num_points, xy_dims)
        padded_labels = padded_contents.view(bs, num_instance * num_points)

        # for padded and output tokens: no pos embed + embedding not a point
        pts_embed = self.label_encoder(padded_labels)
        pos_embed = self.pe_layer.forward_with_coords(
            padded_points, self.input_image_size
        )

        is_mask_token = (
            (padded_labels == EmbeddingIndex.MASK_OUT.value)
            + (padded_labels == EmbeddingIndex.MASK_OUT_1.value)
            + (padded_labels == EmbeddingIndex.MASK_OUT_2.value)
            + (padded_labels == EmbeddingIndex.MASK_OUT_3.value)
        ) > 0
        is_iou_token = padded_labels == EmbeddingIndex.IOU_OUT.value
        is_not_a_point_token = padded_labels == EmbeddingIndex.NOT_A_POINT.value
        is_canvas_token = padded_labels == ExtremeEmbeddingIndex.CANVAS.value
        is_instance_token = padded_labels == ExtremeEmbeddingIndex.INSTANCE.value

        pos_embed[
            is_mask_token
            | is_iou_token
            | is_not_a_point_token
            | is_canvas_token
            | is_instance_token
        ] = 0.0

        pts_embed = pts_embed + pos_embed

        # for SAM we need to have instances indexing
        if with_instance_idx:
            padding_masks = padding_masks.view(bs, num_instance, num_points)
            padded_points = padded_points.view(
                bs, num_instance, num_points, xy_dims
            )  # noqa
            pts_embed = pts_embed.view(bs, num_instance, num_points, -1)
            padded_labels = padded_labels.view(bs, num_instance, num_points)

        pts_embed = self.instance_encoder(pts_embed)

        # # assume padding for the first point indicate if the instance is padded
        # instance_padding_masks = padding_masks[:, :, 0].to(torch.float) * torch.finfo(torch.float).min
        # # (bs, num_inst + num_extra_token, c)
        # instance_embed = self.instance_encoder(pts_embed[:, :, :4], instance_padding_masks)

        # # Extract the instance embeddings and extra tokens separately
        # instance_embed, extra_tokens = \
        #     instance_embed[:, :-self.instance_encoder.num_extra_token], instance_embed[:, -self.instance_encoder.num_extra_token:]

        # # Duplicate instance embeddings to match points dimension
        # instance_embed_expanded = instance_embed.unsqueeze(2).expand(-1, -1, num_points, -1)  # (bs, num_inst, num_points, c)

        # # Update pts_embed by summing it with instance embeddings
        # pts_embed = pts_embed + instance_embed_expanded

        # TODO if we want to include the extra token in the transformer, need to adjust padding etc
        # and also dont forget to remove them before passing by the head

        # padded_labels indicate type of prompt (point, mask, iou..)
        return (
            padding_masks,
            attn_mask,
            padded_points,
            padded_labels,
            pts_embed,
            dense_embeddings,
        )

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size)

    def forward(
        self, batch_data_samples: List[PackPointDetInputs], encode_mask: bool = False
    ):
        """
        Forward pass of the PaddingGenerator.

        Args:
            batch_data_samples (List[PackPointDetInputs]): List of batch data samples.
            encode_mask (bool): Whether to encode mask as prompt.

        Returns:
            tuple: A tuple containing mask padding tensor and points padding tensor (and labels padding tensor if applicable).
        """
        if not batch_data_samples:
            raise ValueError("batch_data_samples should not be empty")

        device = batch_data_samples[0].gt_instances.labels.device
        batch_size = len(batch_data_samples)

        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = (
            unpack_gt_instances(batch_data_samples)
        )  # noqa
        # 1. find max dim in batch (number of prompt, number of instances)
        # NOTE assume only pos point prompts
        max_num_instances: int = 20
        max_num_prompts: int = 2
        # for sample in batch_gt_instances:
        #     n_instances, n_prompts, xy = sample.points.shape
        #     max_num_instances = max(max_num_instances, n_instances)
        #     max_num_prompts = 2

        n_query_per_instances = max_num_prompts + self.n_output_tokens

        # 2. init tensor
        # Create placeholder tensor with zeros with batch dimension
        points_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, 2, device=device
        )

        # Create a mask tensor with ones (ones represent padded input)
        mask_padding_tensor = torch.ones(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        labels_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        # assume no mask_prompt
        dense_embeddings = torch.zeros(
            batch_size,
            max_num_instances,
            self.label_encoder.label_embedding.weight.shape[1],
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        ).to(device)
        # dense_embeddings = dense_embeddings[None, None, :, None, None]
        # dense_embeddings = dense_embeddings.repeat(batch_size, max_num_instances, 1, self.image_embedding_size[0], self.image_embedding_size[1])
        # dense_embeddings = dense_embeddings.expand(batch_size, max_num_instances, -1, self.image_embedding_size[0], self.image_embedding_size[1])  # noqa

        # Fill placeholder and mask tensors
        for idx, sample in enumerate(batch_gt_instances):
            # safeguard empty tensor
            if sample["points"].numel() <= 0:
                continue

            num_instances = sample["points"].size(0)
            num_points = (
                sample["points"].size(1) if len(sample["points"].shape) > 1 else 1
            )  # noqa

            for idx_instance, prompt_type in enumerate(sample["prompt_types"]):

                if prompt_type == PromptType.POINT.value:
                    points_padding[idx, idx_instance, :1] = sample["points"][
                        idx_instance
                    ]

                    # Expand labels to all points
                    labels_padding[idx, idx_instance, :1] = EmbeddingIndex.POS.value
                    labels_padding[idx, idx_instance, 1] = (
                        EmbeddingIndex.NOT_A_POINT.value
                    )

                if prompt_type == PromptType.BOX.value:
                    points_padding[idx, idx_instance, :2] = sample["boxes"][
                        idx_instance
                    ]

                    # Expand labels to all points
                    labels_padding[idx, idx_instance, 0] = (
                        EmbeddingIndex.BOX_CORNER_A.value
                    )
                    labels_padding[idx, idx_instance, 1] = (
                        EmbeddingIndex.BOX_CORNER_B.value
                    )

            # non_init_mask_embed, pos, neg, box_corner_a, box_corner_b
            # labels_padding[labels_padding >= 1] = EmbeddingIndex.POS.value
            # last values are output tokens (n) + Iou
            labels_padding[idx, :num_instances, -self.n_output_tokens] = (
                EmbeddingIndex.MASK_OUT.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 1] = (
                EmbeddingIndex.MASK_OUT_1.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 2] = (
                EmbeddingIndex.MASK_OUT_2.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 3] = (
                EmbeddingIndex.MASK_OUT_3.value
            )  # noqa
            # labels_padding[idx, :num_instances, -self.n_output_tokens:] = EmbeddingIndex.MASK_OUT.value  # noqa
            labels_padding[idx, :num_instances, -1:] = (
                EmbeddingIndex.IOU_OUT.value
            )  # noqa

            # Actual data points are marked as 0
            # mask_padding_tensor[idx, :num_instances, :] = 0
            # mask_padding_tensor[idx, :num_instances, -len(EmbeddingIndex):] = 0  # noqa
            # set output tokens as 0 in mask padding
            # TODO if all is visible, should only set to 0 for the active queries
            mask_padding_tensor[idx, :num_instances, -self.n_output_tokens :] = (
                0  # noqa
            )

            # encode mask proposals (or non-init mask embed) in image-like pos embedding
            if "mask_props" in sample and encode_mask:
                mask_prompts = sample["mask_props"]
                num_instances = mask_prompts.size(0)
                if num_instances == 0:
                    continue

                mask_embeddings = self.mask_downscaling(mask_prompts)
                dense_embeddings[idx, :num_instances] = mask_embeddings
            else:
                dense_embeddings[idx, :num_instances] = (
                    self.label_encoder.label_embedding.weight[
                        EmbeddingIndex.NON_INIT_MASK_EMBED.value
                    ]
                    .view(-1, 1, 1)
                    .repeat(1, dense_embeddings.shape[-2], dense_embeddings.shape[-1])
                )  # noqa

        # 3. compute attention masking
        # 3.1 attention masking, including output token at the end
        attn_mask = self.create_global_attention_mask(
            max_num_instances, max_num_prompts
        )  # noqa
        attn_mask = attn_mask.to(device)

        return self.process_prompt(
            mask_padding_tensor,
            attn_mask,
            points_padding,
            labels_padding.long(),
            dense_embeddings,
        )


@MODELS.register_module()
class ExtremeSAMPaddingGenerator(SAMPaddingGenerator):
    def __init__(
        self,
        embed_dim: int = prompt_embed_dim,
        mask_in_chans: int = 16,
        activation: Type[nn.Module] = nn.GELU,
        image_embedding_size: Tuple[int, int] = (
            image_embedding_size,
            image_embedding_size,
        ),
        input_image_size: Tuple[int, int] = (image_size, image_size),
        label_encoder: OptConfigType = None,
        n_output_tokens: int = 5,  # 3 + 1 (non_ambiguiti mask token) + 1
        use_mask_refinement: bool = False,
        with_instance_embedding: bool = False,
        with_box_corner: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(SAMPaddingGenerator, self).__init__(init_cfg=init_cfg)
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.n_output_tokens = n_output_tokens  # (multimask + iou) tokens
        self.mask_in_chans = mask_in_chans
        self.activation = activation
        self.with_box_corner = with_box_corner

        label_encoder["num_classes"] = len(EmbeddingIndex) + len(
            ExtremeEmbeddingIndex
        )  # non_init_mask_embed, pos, neg, box_corner_a, box_corner_b, not_a_point, 4 mask_outs, iou_out
        label_encoder["embed_dims"] = self.embed_dim
        self.label_encoder = MODELS.build(label_encoder)
        self.use_mask_refinement = use_mask_refinement

        if with_instance_embedding:
            # self.instance_encoder = InstanceEmbedEncoder()
            self.instance_encoder = InstanceEmbedEncoding()
            self.process_prompt = self.process_prompt_with_instance

        self._init_layers()

    def forward(
        self, batch_data_samples: List[PackPointDetInputs], encode_mask: bool = False
    ):
        """
        Forward pass of the PaddingGenerator.

        Args:
            batch_data_samples (List[PackPointDetInputs]): List of batch data samples.
            encode_mask (bool): Whether to encode mask as prompt.

        Returns:
            tuple: A tuple containing mask padding tensor and points padding tensor (and labels padding tensor if applicable).
        """
        if not batch_data_samples:
            raise ValueError("batch_data_samples should not be empty")

        device = batch_data_samples[0].gt_instances.labels.device
        batch_size = len(batch_data_samples)

        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = (
            unpack_gt_instances(batch_data_samples)
        )  # noqa
        # 1. find max dim in batch (number of prompt, number of instances)
        # NOTE assume only pos point prompts
        max_num_instances: int = 28

        if self.with_box_corner:
            max_num_prompts: int = (
                7  # 4 extreme + 2 box + 1 CANVAS (+ 1 instance if 8, otherwise its 7)
            )
        else:
            max_num_prompts: int = (
                5
                # + 3  # 4 extreme + 1 CANVAS (no box corner in the prompt) + 3 for refinement prompts
            )
            # dynamic increase size of prompts...
            if "interactive_points" in batch_gt_instances[0]:
                max_num_prompts += batch_gt_instances[0].interactive_points.shape[1]

        n_query_per_instances = max_num_prompts + self.n_output_tokens

        # 2. init tensor
        # Create placeholder tensor with zeros with batch dimension
        points_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, 2, device=device
        )

        # Create a mask tensor with ones (ones represent padded input)
        mask_padding_tensor = torch.ones(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        labels_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        # assume no mask_prompt
        dense_embeddings = torch.zeros(
            batch_size,
            max_num_instances,
            self.label_encoder.label_embedding.weight.shape[1],
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        ).to(device)

        # Fill placeholder and mask tensors
        for idx, sample in enumerate(batch_gt_instances):
            # safeguard empty tensor
            if sample["anatomical_pole_pools"].numel() <= 0:
                continue

            num_instances = sample["anatomical_pole_pools"].size(0)
            num_points = (
                sample["anatomical_pole_pools"].size(1)
                if len(sample["anatomical_pole_pools"].shape) > 1
                else 1
            )  # noqa

            for idx_instance, _ in enumerate(sample["anatomical_pole_pools"]):
                # 4 extreme points + 2 box corner + CANVAS + INSTANCE
                i = 0
                for extrem_i in [
                    ExtremeEmbeddingIndex.TOP,
                    ExtremeEmbeddingIndex.BOTTOM,
                    ExtremeEmbeddingIndex.LEFT,
                    ExtremeEmbeddingIndex.RIGHT,
                    # NOTE: for ablation
                    # EmbeddingIndex.POS,
                    # EmbeddingIndex.POS,
                    # EmbeddingIndex.POS,
                    # EmbeddingIndex.POS,
                ]:
                    points_padding[idx, idx_instance, i] = sample[
                        "anatomical_pole_pools"
                    ][idx_instance][i]
                    labels_padding[idx, idx_instance, i] = extrem_i.value
                    i = i + 1

                if "interactive_points" in sample:
                    for _ in range(sample.interactive_points.shape[1]):

                        # points_padding
                        points_padding[idx, idx_instance, i] = sample[
                            "interactive_points"
                        ][idx_instance][i - 4]
                        # labels_padding
                        labels_padding[idx, idx_instance, i] = sample[
                            "interactive_points_types"
                        ][idx_instance][i - 4]
                        i = i + 1

                if self.with_box_corner:
                    points_padding[idx, idx_instance, i : i + 2] = sample[
                        "extreme_box"
                    ][idx_instance]
                    labels_padding[idx, idx_instance, i] = (
                        EmbeddingIndex.BOX_CORNER_A.value
                    )
                    labels_padding[idx, idx_instance, i + 1] = (
                        EmbeddingIndex.BOX_CORNER_B.value
                    )

                    labels_padding[idx, idx_instance, i + 2] = (
                        ExtremeEmbeddingIndex.CANVAS.value
                    )
                    mask_padding_tensor[idx, idx_instance, : i + 2 + 1] = 0
                else:
                    labels_padding[idx, idx_instance, i] = (
                        ExtremeEmbeddingIndex.CANVAS.value
                    )
                    mask_padding_tensor[idx, idx_instance, : i + 1] = 0
                    # mask_padding_tensor[idx, idx_instance, :i] = 0

                mask_padding_tensor[idx, idx_instance, -self.n_output_tokens :] = 0

            # non_init_mask_embed, pos, neg, box_corner_a, box_corner_b
            # labels_padding[labels_padding >= 1] = EmbeddingIndex.POS.value
            # last values are output tokens (n) + Iou
            labels_padding[idx, :num_instances, -self.n_output_tokens] = (
                EmbeddingIndex.MASK_OUT.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 1] = (
                EmbeddingIndex.MASK_OUT_1.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 2] = (
                EmbeddingIndex.MASK_OUT_2.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 3] = (
                EmbeddingIndex.MASK_OUT_3.value
            )  # noqa
            # labels_padding[idx, :num_instances, -self.n_output_tokens:] = EmbeddingIndex.MASK_OUT.value  # noqa
            labels_padding[idx, :num_instances, -1:] = (
                EmbeddingIndex.IOU_OUT.value
            )  # noqa

            # encode mask proposals (or non-init mask embed) in image-like pos embedding
            if "mask_props" in sample and encode_mask:
                mask_prompts = sample["mask_props"]
                num_instances = mask_prompts.size(0)
                if num_instances == 0:
                    continue

                mask_embeddings = self.mask_downscaling(mask_prompts)
                dense_embeddings[idx, :num_instances] = mask_embeddings
            else:
                dense_embeddings[idx, :num_instances] = (
                    self.label_encoder.label_embedding.weight[
                        EmbeddingIndex.NON_INIT_MASK_EMBED.value
                    ]
                    .view(-1, 1, 1)
                    .repeat(1, dense_embeddings.shape[-2], dense_embeddings.shape[-1])
                )  # noqa

        # 3. compute attention masking
        # 3.1 attention masking, including output token at the end
        attn_mask = self.create_global_attention_mask(
            max_num_instances, max_num_prompts
        )  # noqa
        attn_mask = attn_mask.to(device)

        return self.process_prompt(
            mask_padding_tensor,
            attn_mask,
            points_padding,
            labels_padding.long(),
            dense_embeddings,
        )


@MODELS.register_module()
class MajMinSAMPaddingGenerator(SAMPaddingGenerator):
    def __init__(
        self,
        embed_dim: int = prompt_embed_dim,
        mask_in_chans: int = 16,
        activation: Type[nn.Module] = nn.GELU,
        image_embedding_size: Tuple[int, int] = (
            image_embedding_size,
            image_embedding_size,
        ),
        input_image_size: Tuple[int, int] = (image_size, image_size),
        label_encoder: OptConfigType = None,
        n_output_tokens: int = 5,  # 3 + 1 (non_ambiguiti mask token) + 1
        use_mask_refinement: bool = False,
        with_box_corner: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(SAMPaddingGenerator, self).__init__(init_cfg=init_cfg)
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.n_output_tokens = n_output_tokens  # (multimask + iou) tokens
        self.mask_in_chans = mask_in_chans
        self.with_box_corner = with_box_corner
        self.activation = activation

        label_encoder["num_classes"] = len(EmbeddingIndex) + len(
            ExtremeEmbeddingIndex
        )  # non_init_mask_embed, pos, neg, box_corner_a, box_corner_b, not_a_point, 4 mask_outs, iou_out
        # label_encoder["num_classes"] = len(EmbeddingIndex)
        label_encoder["embed_dims"] = self.embed_dim
        self.label_encoder = MODELS.build(label_encoder)
        self.use_mask_refinement = use_mask_refinement

        self._init_layers()

    def forward(
        self, batch_data_samples: List[PackPointDetInputs], encode_mask: bool = False
    ):
        """
        Forward pass of the PaddingGenerator.

        Args:
            batch_data_samples (List[PackPointDetInputs]): List of batch data samples.
            encode_mask (bool): Whether to encode mask as prompt.

        Returns:
            tuple: A tuple containing mask padding tensor and points padding tensor (and labels padding tensor if applicable).
        """
        if not batch_data_samples:
            raise ValueError("batch_data_samples should not be empty")

        device = batch_data_samples[0].gt_instances.labels.device
        batch_size = len(batch_data_samples)

        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = (
            unpack_gt_instances(batch_data_samples)
        )  # noqa
        # 1. find max dim in batch (number of prompt, number of instances)
        # NOTE assume only pos point prompts
        max_num_instances: int = 15
        if self.with_box_corner:
            max_num_prompts: int = (
                7  # 4 extreme + 2 box + 1 CANVAS (+ 1 instance if 8, otherwise its 7)
            )
        else:
            max_num_prompts: int = (
                5
                # + 3  # 4 extreme + 1 CANVAS (no box corner in the prompt) + 3 for refinement prompts
            )
            # dynamic increase size of prompts...
            if "interactive_points" in batch_gt_instances[0]:
                max_num_prompts += batch_gt_instances[0].interactive_points.shape[1]

        n_query_per_instances = max_num_prompts + self.n_output_tokens

        # 2. init tensor
        # Create placeholder tensor with zeros with batch dimension
        points_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, 2, device=device
        )

        # Create a mask tensor with ones (ones represent padded input)
        mask_padding_tensor = torch.ones(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        labels_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        # assume no mask_prompt
        dense_embeddings = torch.zeros(
            batch_size,
            max_num_instances,
            self.label_encoder.label_embedding.weight.shape[1],
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        ).to(device)

        # Fill placeholder and mask tensors
        for idx, sample in enumerate(batch_gt_instances):
            # safeguard empty tensor
            if sample["anatomical_pole_pools"].numel() <= 0:
                continue

            num_instances = sample["anatomical_pole_pools"].size(0)
            num_points = (
                sample["anatomical_pole_pools"].size(1)
                if len(sample["anatomical_pole_pools"].shape) > 1
                else 1
            )  # noqa

            for idx_instance, _ in enumerate(sample["anatomical_pole_pools"]):
                # 4 extreme points + 2 box corner + CANVAS + INSTANCE
                i = 0
                for extrem_i in [
                    ExtremeEmbeddingIndex.TOP,
                    ExtremeEmbeddingIndex.TOP,
                    ExtremeEmbeddingIndex.BOTTOM,
                    ExtremeEmbeddingIndex.BOTTOM,
                    # ExtremeEmbeddingIndex.LEFT,
                    # ExtremeEmbeddingIndex.RIGHT,
                ]:
                    points_padding[idx, idx_instance, i] = sample[
                        "anatomical_pole_pools"
                    ][idx_instance][i]
                    labels_padding[idx, idx_instance, i] = extrem_i.value
                    i = i + 1

                if "interactive_points" in sample:
                    for _ in range(sample.interactive_points.shape[1]):

                        # points_padding
                        points_padding[idx, idx_instance, i] = sample[
                            "interactive_points"
                        ][idx_instance][i - 4]
                        # labels_padding
                        labels_padding[idx, idx_instance, i] = sample[
                            "interactive_points_types"
                        ][idx_instance][i - 4]
                        i = i + 1

                if self.with_box_corner:
                    points_padding[idx, idx_instance, i : i + 2] = sample[
                        "extreme_box"
                    ][idx_instance]
                    labels_padding[idx, idx_instance, i] = (
                        EmbeddingIndex.BOX_CORNER_A.value
                    )
                    labels_padding[idx, idx_instance, i + 1] = (
                        EmbeddingIndex.BOX_CORNER_B.value
                    )

                    labels_padding[idx, idx_instance, i + 2] = (
                        ExtremeEmbeddingIndex.CANVAS.value
                    )
                    mask_padding_tensor[idx, idx_instance, : i + 2 + 1] = 0
                else:
                    labels_padding[idx, idx_instance, i] = (
                        ExtremeEmbeddingIndex.CANVAS.value
                    )
                    mask_padding_tensor[idx, idx_instance, : i + 1] = 0

                mask_padding_tensor[idx, idx_instance, -self.n_output_tokens :] = 0

            # non_init_mask_embed, pos, neg, box_corner_a, box_corner_b
            # labels_padding[labels_padding >= 1] = EmbeddingIndex.POS.value
            # last values are output tokens (n) + Iou
            labels_padding[idx, :num_instances, -self.n_output_tokens] = (
                EmbeddingIndex.MASK_OUT.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 1] = (
                EmbeddingIndex.MASK_OUT_1.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 2] = (
                EmbeddingIndex.MASK_OUT_2.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 3] = (
                EmbeddingIndex.MASK_OUT_3.value
            )  # noqa
            # labels_padding[idx, :num_instances, -self.n_output_tokens:] = EmbeddingIndex.MASK_OUT.value  # noqa
            labels_padding[idx, :num_instances, -1:] = (
                EmbeddingIndex.IOU_OUT.value
            )  # noqa

            # encode mask proposals (or non-init mask embed) in image-like pos embedding
            if "mask_props" in sample and encode_mask:
                mask_prompts = sample["mask_props"]
                num_instances = mask_prompts.size(0)
                if num_instances == 0:
                    continue

                mask_embeddings = self.mask_downscaling(mask_prompts)
                dense_embeddings[idx, :num_instances] = mask_embeddings
            else:
                dense_embeddings[idx, :num_instances] = (
                    self.label_encoder.label_embedding.weight[
                        EmbeddingIndex.NON_INIT_MASK_EMBED.value
                    ]
                    .view(-1, 1, 1)
                    .repeat(1, dense_embeddings.shape[-2], dense_embeddings.shape[-1])
                )  # noqa

        # 3. compute attention masking
        # 3.1 attention masking, including output token at the end
        attn_mask = self.create_global_attention_mask(
            max_num_instances, max_num_prompts
        )  # noqa
        attn_mask = attn_mask.to(device)

        return self.process_prompt(
            mask_padding_tensor,
            attn_mask,
            points_padding,
            labels_padding.long(),
            dense_embeddings,
        )


@MODELS.register_module()
class OnlyMajExtremeSAMPaddingGenerator(MajMinSAMPaddingGenerator):
    def forward(
        self, batch_data_samples: List[PackPointDetInputs], encode_mask: bool = False
    ):
        """
        Forward pass of the PaddingGenerator.

        Args:
            batch_data_samples (List[PackPointDetInputs]): List of batch data samples.
            encode_mask (bool): Whether to encode mask as prompt.

        Returns:
            tuple: A tuple containing mask padding tensor and points padding tensor (and labels padding tensor if applicable).
        """
        if not batch_data_samples:
            raise ValueError("batch_data_samples should not be empty")

        device = batch_data_samples[0].gt_instances.labels.device
        batch_size = len(batch_data_samples)

        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = (
            unpack_gt_instances(batch_data_samples)
        )  # noqa
        # 1. find max dim in batch (number of prompt, number of instances)
        # NOTE assume only pos point prompts
        max_num_instances: int = 15
        if self.with_box_corner:
            max_num_prompts: int = (
                7  # 4 extreme + 2 box + 1 CANVAS (+ 1 instance if 8, otherwise its 7)
            )
        else:
            max_num_prompts: int = (
                3
                # + 3  # 4 extreme + 1 CANVAS (no box corner in the prompt) + 3 for refinement prompts
            )
            # dynamic increase size of prompts...
            if "interactive_points" in batch_gt_instances[0]:
                max_num_prompts += batch_gt_instances[0].interactive_points.shape[1]

        n_query_per_instances = max_num_prompts + self.n_output_tokens

        # 2. init tensor
        # Create placeholder tensor with zeros with batch dimension
        points_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, 2, device=device
        )

        # Create a mask tensor with ones (ones represent padded input)
        mask_padding_tensor = torch.ones(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        labels_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        # assume no mask_prompt
        dense_embeddings = torch.zeros(
            batch_size,
            max_num_instances,
            self.label_encoder.label_embedding.weight.shape[1],
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        ).to(device)

        # Fill placeholder and mask tensors
        for idx, sample in enumerate(batch_gt_instances):
            # safeguard empty tensor
            if sample["anatomical_pole_pools"].numel() <= 0:
                continue

            num_instances = sample["anatomical_pole_pools"].size(0)
            num_points = (
                sample["anatomical_pole_pools"].size(1)
                if len(sample["anatomical_pole_pools"].shape) > 1
                else 1
            )  # noqa

            for idx_instance, _ in enumerate(sample["anatomical_pole_pools"]):
                # 4 extreme points + 2 box corner + CANVAS + INSTANCE
                i = 0
                for extrem_i in [
                    ExtremeEmbeddingIndex.TOP,
                    ExtremeEmbeddingIndex.TOP,
                    # ExtremeEmbeddingIndex.BOTTOM,
                    # ExtremeEmbeddingIndex.BOTTOM,
                    # ExtremeEmbeddingIndex.LEFT,
                    # ExtremeEmbeddingIndex.RIGHT,
                ]:
                    points_padding[idx, idx_instance, i] = sample[
                        "anatomical_pole_pools"
                    ][idx_instance][i]
                    labels_padding[idx, idx_instance, i] = extrem_i.value
                    i = i + 1

                if "interactive_points" in sample:
                    for _ in range(sample.interactive_points.shape[1]):

                        # points_padding
                        points_padding[idx, idx_instance, i] = sample[
                            "interactive_points"
                        ][idx_instance][i - 2]
                        # labels_padding
                        labels_padding[idx, idx_instance, i] = sample[
                            "interactive_points_types"
                        ][idx_instance][i - 2]
                        i = i + 1

                if self.with_box_corner:
                    points_padding[idx, idx_instance, i : i + 2] = sample[
                        "extreme_box"
                    ][idx_instance]
                    labels_padding[idx, idx_instance, i] = (
                        EmbeddingIndex.BOX_CORNER_A.value
                    )
                    labels_padding[idx, idx_instance, i + 1] = (
                        EmbeddingIndex.BOX_CORNER_B.value
                    )

                    labels_padding[idx, idx_instance, i + 2] = (
                        ExtremeEmbeddingIndex.CANVAS.value
                    )
                    mask_padding_tensor[idx, idx_instance, : i + 2 + 1] = 0
                else:
                    labels_padding[idx, idx_instance, i] = (
                        ExtremeEmbeddingIndex.CANVAS.value
                    )
                    mask_padding_tensor[idx, idx_instance, : i + 1] = 0

                mask_padding_tensor[idx, idx_instance, -self.n_output_tokens :] = 0

            # non_init_mask_embed, pos, neg, box_corner_a, box_corner_b
            # labels_padding[labels_padding >= 1] = EmbeddingIndex.POS.value
            # last values are output tokens (n) + Iou
            labels_padding[idx, :num_instances, -self.n_output_tokens] = (
                EmbeddingIndex.MASK_OUT.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 1] = (
                EmbeddingIndex.MASK_OUT_1.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 2] = (
                EmbeddingIndex.MASK_OUT_2.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 3] = (
                EmbeddingIndex.MASK_OUT_3.value
            )  # noqa
            # labels_padding[idx, :num_instances, -self.n_output_tokens:] = EmbeddingIndex.MASK_OUT.value  # noqa
            labels_padding[idx, :num_instances, -1:] = (
                EmbeddingIndex.IOU_OUT.value
            )  # noqa

            # encode mask proposals (or non-init mask embed) in image-like pos embedding
            if "mask_props" in sample and encode_mask:
                mask_prompts = sample["mask_props"]
                num_instances = mask_prompts.size(0)
                if num_instances == 0:
                    continue

                mask_embeddings = self.mask_downscaling(mask_prompts)
                dense_embeddings[idx, :num_instances] = mask_embeddings
            else:
                dense_embeddings[idx, :num_instances] = (
                    self.label_encoder.label_embedding.weight[
                        EmbeddingIndex.NON_INIT_MASK_EMBED.value
                    ]
                    .view(-1, 1, 1)
                    .repeat(1, dense_embeddings.shape[-2], dense_embeddings.shape[-1])
                )  # noqa

        # 3. compute attention masking
        # 3.1 attention masking, including output token at the end
        attn_mask = self.create_global_attention_mask(
            max_num_instances, max_num_prompts
        )  # noqa
        attn_mask = attn_mask.to(device)

        return self.process_prompt(
            mask_padding_tensor,
            attn_mask,
            points_padding,
            labels_padding.long(),
            dense_embeddings,
        )


@MODELS.register_module()
class PointSAMPaddingGenerator(MajMinSAMPaddingGenerator):
    def forward(
        self, batch_data_samples: List[PackPointDetInputs], encode_mask: bool = False
    ):
        """
        Forward pass of the PaddingGenerator.

        Args:
            batch_data_samples (List[PackPointDetInputs]): List of batch data samples.
            encode_mask (bool): Whether to encode mask as prompt.

        Returns:
            tuple: A tuple containing mask padding tensor and points padding tensor (and labels padding tensor if applicable).
        """
        if not batch_data_samples:
            raise ValueError("batch_data_samples should not be empty")

        device = batch_data_samples[0].gt_instances.labels.device
        batch_size = len(batch_data_samples)

        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = (
            unpack_gt_instances(batch_data_samples)
        )  # noqa
        # 1. find max dim in batch (number of prompt, number of instances)
        # NOTE assume only pos point prompts
        max_num_instances: int = 14
        if self.with_box_corner:
            max_num_prompts: int = (
                7  # 4 extreme + 2 box + 1 CANVAS (+ 1 instance if 8, otherwise its 7)
            )
        else:
            max_num_prompts: int = (
                1
                # + 3  # 4 extreme + 1 CANVAS (no box corner in the prompt) + 3 for refinement prompts
            )
            # dynamic increase size of prompts...
            if "interactive_points" in batch_gt_instances[0]:
                max_num_prompts += batch_gt_instances[0].interactive_points.shape[1]

        n_query_per_instances = max_num_prompts + self.n_output_tokens

        # 2. init tensor
        # Create placeholder tensor with zeros with batch dimension
        points_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, 2, device=device
        )

        # Create a mask tensor with ones (ones represent padded input)
        mask_padding_tensor = torch.ones(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        labels_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        # assume no mask_prompt
        dense_embeddings = torch.zeros(
            batch_size,
            max_num_instances,
            self.label_encoder.label_embedding.weight.shape[1],
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        ).to(device)

        # Fill placeholder and mask tensors
        for idx, sample in enumerate(batch_gt_instances):
            # safeguard empty tensor
            if sample["anatomical_pole_pools"].numel() <= 0:
                continue

            num_instances = sample["anatomical_pole_pools"].size(0)
            num_points = (
                sample["anatomical_pole_pools"].size(1)
                if len(sample["anatomical_pole_pools"].shape) > 1
                else 1
            )  # noqa

            for idx_instance, _ in enumerate(sample["anatomical_pole_pools"]):
                # 4 extreme points + 2 box corner + CANVAS + INSTANCE
                i = 0
                for extrem_i in [EmbeddingIndex.POS]:
                    points_padding[idx, idx_instance, i] = sample["points"][
                        idx_instance
                    ][i]
                    labels_padding[idx, idx_instance, i] = extrem_i.value
                    i = i + 1
                    # labels_padding[idx, idx_instance, 1] = (
                    #     EmbeddingIndex.NOT_A_POINT.value
                    # )
                    # i = i + 1

                if "interactive_points" in sample:
                    for _ in range(sample.interactive_points.shape[1]):

                        # points_padding
                        points_padding[idx, idx_instance, i] = sample[
                            "interactive_points"
                        ][idx_instance][i - 2]
                        # labels_padding
                        labels_padding[idx, idx_instance, i] = sample[
                            "interactive_points_types"
                        ][idx_instance][i - 2]
                        i = i + 1

                if self.with_box_corner:
                    points_padding[idx, idx_instance, i : i + 2] = sample[
                        "extreme_box"
                    ][idx_instance]
                    labels_padding[idx, idx_instance, i] = (
                        EmbeddingIndex.BOX_CORNER_A.value
                    )
                    labels_padding[idx, idx_instance, i + 1] = (
                        EmbeddingIndex.BOX_CORNER_B.value
                    )

                    labels_padding[idx, idx_instance, i + 2] = (
                        ExtremeEmbeddingIndex.CANVAS.value
                    )
                    mask_padding_tensor[idx, idx_instance, : i + 2 + 1] = 0
                else:
                    # labels_padding[idx, idx_instance, i] = (
                    #     ExtremeEmbeddingIndex.CANVAS.value
                    # )
                    # mask_padding_tensor[idx, idx_instance, : i + 1] = 0
                    mask_padding_tensor[idx, idx_instance, :i] = 0

                mask_padding_tensor[idx, idx_instance, -self.n_output_tokens :] = 0

            # non_init_mask_embed, pos, neg, box_corner_a, box_corner_b
            # labels_padding[labels_padding >= 1] = EmbeddingIndex.POS.value
            # last values are output tokens (n) + Iou
            labels_padding[idx, :num_instances, -self.n_output_tokens] = (
                EmbeddingIndex.MASK_OUT.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 1] = (
                EmbeddingIndex.MASK_OUT_1.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 2] = (
                EmbeddingIndex.MASK_OUT_2.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 3] = (
                EmbeddingIndex.MASK_OUT_3.value
            )  # noqa
            # labels_padding[idx, :num_instances, -self.n_output_tokens:] = EmbeddingIndex.MASK_OUT.value  # noqa
            labels_padding[idx, :num_instances, -1:] = (
                EmbeddingIndex.IOU_OUT.value
            )  # noqa

            # encode mask proposals (or non-init mask embed) in image-like pos embedding
            if "mask_props" in sample and encode_mask:
                mask_prompts = sample["mask_props"]
                num_instances = mask_prompts.size(0)
                if num_instances == 0:
                    continue

                mask_embeddings = self.mask_downscaling(mask_prompts)
                dense_embeddings[idx, :num_instances] = mask_embeddings
            else:
                dense_embeddings[idx, :num_instances] = (
                    self.label_encoder.label_embedding.weight[
                        EmbeddingIndex.NON_INIT_MASK_EMBED.value
                    ]
                    .view(-1, 1, 1)
                    .repeat(1, dense_embeddings.shape[-2], dense_embeddings.shape[-1])
                )  # noqa

        # 3. compute attention masking
        # 3.1 attention masking, including output token at the end
        attn_mask = self.create_global_attention_mask(
            max_num_instances, max_num_prompts
        )  # noqa
        attn_mask = attn_mask.to(device)

        return self.process_prompt(
            mask_padding_tensor,
            attn_mask,
            points_padding,
            labels_padding.long(),
            dense_embeddings,
        )


@MODELS.register_module()
class MixedSAMPaddingGenerator(MajMinSAMPaddingGenerator):
    def forward(
        self, batch_data_samples: List[PackPointDetInputs], encode_mask: bool = False
    ):
        """
        Forward pass of the PaddingGenerator.

        Args:
            batch_data_samples (List[PackPointDetInputs]): List of batch data samples.
            encode_mask (bool): Whether to encode mask as prompt.

        Returns:
            tuple: A tuple containing mask padding tensor and points padding tensor (and labels padding tensor if applicable).
        """
        if not batch_data_samples:
            raise ValueError("batch_data_samples should not be empty")

        device = batch_data_samples[0].gt_instances.labels.device
        batch_size = len(batch_data_samples)

        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = (
            unpack_gt_instances(batch_data_samples)
        )  # noqa
        # 1. find max dim in batch (number of prompt, number of instances)
        # NOTE assume only pos point prompts
        max_num_instances: int = 15
        if self.with_box_corner:
            max_num_prompts: int = (
                7  # 4 extreme + 2 box + 1 CANVAS (+ 1 instance if 8, otherwise its 7)
            )
        else:
            max_num_prompts: int = (
                5
                # + 3  # 4 extreme + 1 CANVAS (no box corner in the prompt) + 3 for refinement prompts
            )
            # dynamic increase size of prompts...
            if "interactive_points" in batch_gt_instances[0]:
                max_num_prompts += batch_gt_instances[0].interactive_points.shape[1]

        n_query_per_instances = max_num_prompts + self.n_output_tokens

        # 2. init tensor
        # Create placeholder tensor with zeros with batch dimension
        points_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, 2, device=device
        )

        # Create a mask tensor with ones (ones represent padded input)
        mask_padding_tensor = torch.ones(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        labels_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        # assume no mask_prompt
        dense_embeddings = torch.zeros(
            batch_size,
            max_num_instances,
            self.label_encoder.label_embedding.weight.shape[1],
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        ).to(device)

        # Fill placeholder and mask tensors
        for idx, sample in enumerate(batch_gt_instances):
            # safeguard empty tensor
            if sample["anatomical_pole_pools"].numel() <= 0:
                continue

            num_instances = sample["anatomical_pole_pools"].size(0)
            num_points = (
                sample["anatomical_pole_pools"].size(1)
                if len(sample["anatomical_pole_pools"].shape) > 1
                else 1
            )  # noqa

            for idx_instance, _ in enumerate(sample["prompt_types"]):
                # 4 extreme points + 2 box corner + CANVAS + INSTANCE
                i = 0

                # print(sample["prompt_types"], sample["prompt_types"][idx_instance])
                if sample["prompt_types"][idx_instance] == PromptType.EXTREME.value:
                    embed_index = [
                        ExtremeEmbeddingIndex.TOP,
                        ExtremeEmbeddingIndex.BOTTOM,
                        ExtremeEmbeddingIndex.LEFT,
                        ExtremeEmbeddingIndex.RIGHT,
                    ]
                    sample_key = "anatomical_pole_pools"
                    # print("EXTREME")
                elif sample["prompt_types"][idx_instance] == PromptType.MAJMIN.value:
                    embed_index = [
                        ExtremeEmbeddingIndex.MAJ,
                        ExtremeEmbeddingIndex.MAJ,
                        ExtremeEmbeddingIndex.MIN,
                        ExtremeEmbeddingIndex.MIN,
                    ]
                    sample_key = "anatomical_pole_pools"
                    # print("MAJMIN")
                elif sample["prompt_types"][idx_instance] == PromptType.POINT.value:
                    embed_index = [
                        EmbeddingIndex.POS,
                        EmbeddingIndex.NOT_A_POINT,
                        EmbeddingIndex.NOT_A_POINT,
                        EmbeddingIndex.NOT_A_POINT,
                    ]
                    sample_key = "anatomical_pole_pools"
                elif sample["prompt_types"][idx_instance] == PromptType.BOX.value:
                    embed_index = [
                        EmbeddingIndex.BOX_CORNER_A,
                        EmbeddingIndex.BOX_CORNER_B,
                        EmbeddingIndex.NOT_A_POINT,
                        EmbeddingIndex.NOT_A_POINT,
                    ]
                    sample_key = "anatomical_pole_pools"

                for extrem_i in embed_index:
                    points_padding[idx, idx_instance, i] = sample[sample_key][
                        idx_instance
                    ][i]
                    labels_padding[idx, idx_instance, i] = extrem_i.value
                    i = i + 1

                if "interactive_points" in sample:
                    for _ in range(sample.interactive_points.shape[1]):

                        # points_padding
                        points_padding[idx, idx_instance, i] = sample[
                            "interactive_points"
                        ][idx_instance][i - len(embed_index)]
                        # labels_padding
                        labels_padding[idx, idx_instance, i] = sample[
                            "interactive_points_types"
                        ][idx_instance][i - len(embed_index)]
                        i = i + 1

                if self.with_box_corner:
                    points_padding[idx, idx_instance, i : i + 2] = sample[
                        "extreme_box"
                    ][idx_instance]
                    labels_padding[idx, idx_instance, i] = (
                        EmbeddingIndex.BOX_CORNER_A.value
                    )
                    labels_padding[idx, idx_instance, i + 1] = (
                        EmbeddingIndex.BOX_CORNER_B.value
                    )

                    labels_padding[idx, idx_instance, i + 2] = (
                        ExtremeEmbeddingIndex.CANVAS.value
                    )
                    mask_padding_tensor[idx, idx_instance, : i + 2 + 1] = 0
                else:
                    labels_padding[idx, idx_instance, i] = (
                        ExtremeEmbeddingIndex.CANVAS.value
                    )
                    mask_padding_tensor[idx, idx_instance, : i + 1] = 0

                mask_padding_tensor[idx, idx_instance, -self.n_output_tokens :] = 0

            # non_init_mask_embed, pos, neg, box_corner_a, box_corner_b
            # labels_padding[labels_padding >= 1] = EmbeddingIndex.POS.value
            # last values are output tokens (n) + Iou
            labels_padding[idx, :num_instances, -self.n_output_tokens] = (
                EmbeddingIndex.MASK_OUT.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 1] = (
                EmbeddingIndex.MASK_OUT_1.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 2] = (
                EmbeddingIndex.MASK_OUT_2.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 3] = (
                EmbeddingIndex.MASK_OUT_3.value
            )  # noqa
            # labels_padding[idx, :num_instances, -self.n_output_tokens:] = EmbeddingIndex.MASK_OUT.value  # noqa
            labels_padding[idx, :num_instances, -1:] = (
                EmbeddingIndex.IOU_OUT.value
            )  # noqa

            # encode mask proposals (or non-init mask embed) in image-like pos embedding
            if "mask_props" in sample and encode_mask:
                mask_prompts = sample["mask_props"]
                num_instances = mask_prompts.size(0)
                if num_instances == 0:
                    continue

                mask_embeddings = self.mask_downscaling(mask_prompts)
                dense_embeddings[idx, :num_instances] = mask_embeddings
            else:
                dense_embeddings[idx, :num_instances] = (
                    self.label_encoder.label_embedding.weight[
                        EmbeddingIndex.NON_INIT_MASK_EMBED.value
                    ]
                    .view(-1, 1, 1)
                    .repeat(1, dense_embeddings.shape[-2], dense_embeddings.shape[-1])
                )  # noqa

        # 3. compute attention masking
        # 3.1 attention masking, including output token at the end
        attn_mask = self.create_global_attention_mask(
            max_num_instances, max_num_prompts
        )  # noqa
        attn_mask = attn_mask.to(device)

        return self.process_prompt(
            mask_padding_tensor,
            attn_mask,
            points_padding,
            labels_padding.long(),
            dense_embeddings,
        )


@MODELS.register_module()
class BoxSAMPaddingGenerator(MajMinSAMPaddingGenerator):
    def __init__(
        self,
        embed_dim: int = prompt_embed_dim,
        mask_in_chans: int = 16,
        activation: Type[nn.Module] = nn.GELU,
        image_embedding_size: Tuple[int, int] = (
            image_embedding_size,
            image_embedding_size,
        ),
        input_image_size: Tuple[int, int] = (image_size, image_size),
        label_encoder: OptConfigType = None,
        n_output_tokens: int = 5,  # 3 + 1 (non_ambiguiti mask token) + 1
        use_mask_refinement: bool = False,
        with_box_corner: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(SAMPaddingGenerator, self).__init__(init_cfg=init_cfg)
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.n_output_tokens = n_output_tokens  # (multimask + iou) tokens
        self.mask_in_chans = mask_in_chans
        self.with_box_corner = with_box_corner
        self.activation = activation

        label_encoder["num_classes"] = len(EmbeddingIndex)
        label_encoder["embed_dims"] = self.embed_dim
        self.label_encoder = MODELS.build(label_encoder)
        self.use_mask_refinement = use_mask_refinement

        self._init_layers()

    def forward(
        self, batch_data_samples: List[PackPointDetInputs], encode_mask: bool = False
    ):
        """
        Forward pass of the PaddingGenerator.

        Args:
            batch_data_samples (List[PackPointDetInputs]): List of batch data samples.
            encode_mask (bool): Whether to encode mask as prompt.

        Returns:
            tuple: A tuple containing mask padding tensor and points padding tensor (and labels padding tensor if applicable).
        """
        if not batch_data_samples:
            raise ValueError("batch_data_samples should not be empty")

        device = batch_data_samples[0].gt_instances.labels.device
        batch_size = len(batch_data_samples)

        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = (
            unpack_gt_instances(batch_data_samples)
        )  # noqa
        # 1. find max dim in batch (number of prompt, number of instances)
        # NOTE assume only pos point prompts
        max_num_instances: int = 15
        if self.with_box_corner:
            max_num_prompts: int = (
                7  # 4 extreme + 2 box + 1 CANVAS (+ 1 instance if 8, otherwise its 7)
            )
        else:
            max_num_prompts: int = (
                2
                # + 3  # 4 extreme + 1 CANVAS (no box corner in the prompt) + 3 for refinement prompts
            )

        n_query_per_instances = max_num_prompts + self.n_output_tokens

        # 2. init tensor
        # Create placeholder tensor with zeros with batch dimension
        points_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, 2, device=device
        )

        # Create a mask tensor with ones (ones represent padded input)
        mask_padding_tensor = torch.ones(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        labels_padding = torch.zeros(
            batch_size, max_num_instances, n_query_per_instances, device=device
        )

        # assume no mask_prompt
        dense_embeddings = torch.zeros(
            batch_size,
            max_num_instances,
            self.label_encoder.label_embedding.weight.shape[1],
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        ).to(device)

        # Fill placeholder and mask tensors
        for idx, sample in enumerate(batch_gt_instances):
            # safeguard empty tensor
            if sample["anatomical_pole_pools"].numel() <= 0:
                continue

            num_instances = sample["anatomical_pole_pools"].size(0)
            num_points = (
                sample["anatomical_pole_pools"].size(1)
                if len(sample["anatomical_pole_pools"].shape) > 1
                else 1
            )  # noqa

            for idx_instance, _ in enumerate(sample["anatomical_pole_pools"]):
                # 4 extreme points + 2 box corner + CANVAS + INSTANCE
                i = 0
                for extrem_i in [
                    EmbeddingIndex.BOX_CORNER_A,
                    EmbeddingIndex.BOX_CORNER_B,
                ]:
                    points_padding[idx, idx_instance, i] = sample[
                        "anatomical_pole_pools"
                    ][idx_instance][i]
                    labels_padding[idx, idx_instance, i] = extrem_i.value
                    i = i + 1

                # labels_padding[idx, idx_instance, 0] = EmbeddingIndex.BOX_CORNER_A.value
                # labels_padding[idx, idx_instance, 1] = EmbeddingIndex.BOX_CORNER_B.value

                mask_padding_tensor[idx, idx_instance, :i] = 0

                mask_padding_tensor[idx, idx_instance, -self.n_output_tokens :] = 0

            # non_init_mask_embed, pos, neg, box_corner_a, box_corner_b
            # labels_padding[labels_padding >= 1] = EmbeddingIndex.POS.value
            # last values are output tokens (n) + Iou
            labels_padding[idx, :num_instances, -self.n_output_tokens] = (
                EmbeddingIndex.MASK_OUT.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 1] = (
                EmbeddingIndex.MASK_OUT_1.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 2] = (
                EmbeddingIndex.MASK_OUT_2.value
            )  # noqa
            labels_padding[idx, :num_instances, -self.n_output_tokens + 3] = (
                EmbeddingIndex.MASK_OUT_3.value
            )  # noqa
            # labels_padding[idx, :num_instances, -self.n_output_tokens:] = EmbeddingIndex.MASK_OUT.value  # noqa
            labels_padding[idx, :num_instances, -1:] = (
                EmbeddingIndex.IOU_OUT.value
            )  # noqa

            # encode mask proposals (or non-init mask embed) in image-like pos embedding
            if "mask_props" in sample and encode_mask:
                mask_prompts = sample["mask_props"]
                num_instances = mask_prompts.size(0)
                if num_instances == 0:
                    continue

                mask_embeddings = self.mask_downscaling(mask_prompts)
                dense_embeddings[idx, :num_instances] = mask_embeddings
            else:
                dense_embeddings[idx, :num_instances] = (
                    self.label_encoder.label_embedding.weight[
                        EmbeddingIndex.NON_INIT_MASK_EMBED.value
                    ]
                    .view(-1, 1, 1)
                    .repeat(1, dense_embeddings.shape[-2], dense_embeddings.shape[-1])
                )  # noqa

        # 3. compute attention masking
        # 3.1 attention masking, including output token at the end
        attn_mask = self.create_global_attention_mask(
            max_num_instances, max_num_prompts
        )  # noqa
        attn_mask = attn_mask.to(device)

        return self.process_prompt(
            mask_padding_tensor,
            attn_mask,
            points_padding,
            labels_padding.long(),
            dense_embeddings,
        )
