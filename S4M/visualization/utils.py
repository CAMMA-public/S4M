import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Optional, Tuple, Union
import os


def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(
            unbiased, q=self.n_components, center=False, niter=4
        )
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def pca(
    image_feats_list: List[torch.Tensor],  # Each tensor: (B, C, H, W)
    dim: int = 3,
    fit_pca: Optional[Union["TorchPCA", "PCA"]] = None,
    use_torch_pca: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[List[torch.Tensor], Union["TorchPCA", "PCA"]]:
    """
    Apply PCA on a list of 4D tensors (B, C, H, W).

    Returns:
        reduced_feats: List of reduced tensors with shape (B, dim, H, W)
        fit_pca: The fitted PCA object
    """
    for feats in image_feats_list:
        if feats.ndim != 4:
            raise ValueError(
                f"Each tensor must have shape (B, C, H, W), got {feats.shape}"
            )

    device = image_feats_list[0].device

    def flatten(
        tensor: torch.Tensor, target_size: Optional[int] = None
    ) -> torch.Tensor:
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return (
            tensor.permute(1, 0, 2, 3)
            .reshape(C, B * H * W)
            .permute(1, 0)
            .detach()
            .cpu()
        )

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = [flatten(feats, target_size) for feats in image_feats_list]
    x = torch.cat(flattened_feats, dim=0)

    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            from sklearn.decomposition import PCA

            fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca


@torch.no_grad()
def plot_feats(
    image: torch.Tensor,
    encoder_featMap: torch.Tensor,
    decoder_featMap: torch.Tensor,
    save_path: str,
):
    [encoder_featMap_pca, decoder_featMap_pca], _ = pca(
        [encoder_featMap, decoder_featMap]
    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(normalize_to_0_1(image).permute(1, 2, 0).detach().cpu())
    ax[0].set_title("Image")
    ax[1].imshow(encoder_featMap_pca[0].permute(1, 2, 0).detach().cpu())
    ax[1].set_title("Encoder Features")
    ax[2].imshow(decoder_featMap_pca[0].permute(1, 2, 0).detach().cpu())
    ax[2].set_title("Decoder Features")
    remove_axes(ax)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def unnormalize_imagenet(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(-1, 1, 1)
    return t * std + mean


def normalize_to_0_1(t: torch.Tensor) -> torch.Tensor:
    t_min = t.min()
    t_max = t.max()
    return (t - t_min) / (t_max - t_min + 1e-5)


# plot_feats(
#     unnormalize_imagenet(batch_inputs[0]),
#     img_feats[-1],
#     image_embedding,
#     f"./outputs/vis/{batch_img_metas[0]['img_id']}.png",
# )
