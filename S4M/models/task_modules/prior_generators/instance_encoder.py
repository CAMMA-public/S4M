from typing import Dict, List, Tuple, Union, Optional

from torch import nn, Tensor
import torch
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig
from mmcv.cnn.bricks.transformer import FFN
from mmdet.models.layers.transformer import DetrTransformerEncoder, MLP


@MODELS.register_module()
class InstanceEmbedEncoding(BaseModule):
    """
    Encodes instance into embedding space, useful to link multiple prompts together.
    """

    def __init__(self,
                 embed_dim: int = 256,
                 num_instances: int = 20,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg)
        self.embed_dim = embed_dim
        self.num_instances = num_instances
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize the embedding layer."""
        self.instance_embedding = nn.Embedding(self.num_instances*13, self.embed_dim )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (bs, num_inst, num_prompt, c).
            key_padding_mask (Optional[Tensor]): Mask tensor.

        Returns:
            Tensor: Encoded tensor.
        """
        bs, num_inst, num_prompt, c = x.shape

        # [num_instances * N, C] -> [num_instances, N, C]
        instance_embed = self.instance_embedding.weight.view(self.num_instances, num_prompt, -1)

        # Shuffle on num_instances
        # perm = torch.randperm(self.num_instances, device=instance_embed.device)
        # instance_embed = instance_embed[perm]

        instance_embed = instance_embed.unsqueeze(0).expand(bs, -1, -1, -1)
        return x + instance_embed



@MODELS.register_module()
class InstanceEmbedEncoder(BaseModule):
    """
    Encodes instance into embedding space, usefull to link multiple prompt together
    """

    def __init__(self,
                 encoder_config=dict(  # DetrTransformerEncoder
                    num_layers=2,
                    layer_cfg=dict(  # DetrTransformerEncoderLayer
                        self_attn_cfg=dict(  # MultiheadAttention
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0,
                            batch_first=True),
                        ffn_cfg=dict(
                            embed_dims=256,
                            feedforward_channels=2048,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True)
                        )
                    )
                 ),
                 mlp_config=dict(
                    input_dim=256*4,
                    hidden_dim=256,
                    output_dim=256,
                    num_layers=2,
                 ),
                 num_extra_token=1,  # scene embedding
                 init_cfg: OptMultiConfig = None) -> None:
        """
        Initialize the LabelEncoder.

        Args:
            num_classes (int): Number of classes. Default: 5.
              for pos, neg, box_corner_a, box_corner_b, not_a_point
            embed_dims (int): Dimension of the embedding. Default: 256.
        """
        super().__init__(init_cfg)
        self.encoder_config = encoder_config
        self.mlp_config = mlp_config
        self.num_extra_token = num_extra_token
        self.embed_dims = encoder_config['layer_cfg']['self_attn_cfg']['embed_dims']
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize the embedding layer."""
        self.encoder = DetrTransformerEncoder(**self.encoder_config)
        self.mlp = MLP(**self.mlp_config)
        self.extra_tokens = nn.Parameter(
            torch.zeros(1, self.num_extra_token, self.embed_dims))

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.xavier_uniform_(self.extra_tokens)


    # def forward(self, x: Tensor, key_padding_mask: Tensor) -> Tensor:
    #     """
    #     input x shape: (bs, num_inst, num_prompt, c)
    #     output shape: (bs, num_inst + num_extra_token, c)
    #     """
    #     bs, num_inst, num_prompt, c = x.shape

    #     # Flatten prompts (bs, num_inst, num_prompt * c)
    #     x = x.view(bs, num_inst, num_prompt * c)

    #     # Pass through MLP (bs, num_inst, c)
    #     x = self.mlp(x)

    #     return x

    def forward(self, x: Tensor, key_padding_mask: Tensor) -> Tensor:
        """
        input x shape: (bs, num_inst, num_prompt, c)
        output shape: (bs, num_inst + num_extra_token, c)
        """
        bs, num_inst, num_prompt, c = x.shape

        # Flatten prompts (bs, num_inst, num_prompt * c)
        x = x.view(bs, num_inst, num_prompt * c)

        # Pass through MLP (bs, num_inst, c)
        x = self.mlp(x)

        # Expand extra tokens to batch size (bs, num_extra_token, c)
        extra_tokens = self.extra_tokens.expand(bs, -1, -1)

        # Concatenate extra tokens (bs, num_inst + num_extra_token, c)
        x = torch.cat([x, extra_tokens], dim=1)

        # Create padding mask for extra tokens (assumed to be False, meaning they are not masked)
        extra_padding_mask = torch.zeros(bs, self.num_extra_token, device=key_padding_mask.device, dtype=key_padding_mask.dtype)

        # Concatenate the original padding mask with the extra padding mask
        key_padding_mask = torch.cat([key_padding_mask, extra_padding_mask], dim=1)

        # Pass through encoder
        # need to pass padding mask
        x = self.encoder(x, None, key_padding_mask)

        return x


if __name__ == "__main__":
    # Test script
    model = InstanceEmbedEncoder(num_extra_token=100)
    model.init_weights()
    test_input = torch.randn(2, 5, 4, 256)  # (batch_size, num_inst, num_prompt, embed_dims) # noqa
    output = model(test_input)
    print("Output shape:", output.shape)
