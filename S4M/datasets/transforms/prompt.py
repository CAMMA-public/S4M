from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.registry import TRANSFORMS
import numpy as np
from sklearn.decomposition import PCA
import random
import torch
from typing import List, Optional, Sequence, Tuple, Union, Dict
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import cv2
import matplotlib.cm as cm
from PIL import Image, ImageDraw
from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset
import os.path as osp
from typing import Literal
from mmcv.transforms.utils import cache_randomness

# from scipy.spatial import distance_matrix

# from skimage.graph import route_through_array
from skimage.morphology import skeletonize


# Define the Enum for prompt types
class PromptType(Enum):
    POINT = 0
    BOX = 1
    EXTREME = 2
    MAJMIN = 3


@TRANSFORMS.register_module()
class GetEdgeMask(BaseTransform):
    def __init__(self, erosion: int = 3, edge_type: str = "inner"):
        assert edge_type in {
            "inner",
            "outer",
            "both",
        }, "edge_type must be 'inner', 'outer', or 'both'"  # noqa
        self.erosion = erosion
        self.edge_type = edge_type

    def compute_edge_mask(self, mask: np.ndarray) -> dict:
        kernel = np.ones((self.erosion, self.erosion), dtype=np.uint8)

        eroded_mask = cv2.erode(
            mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        dilated_mask = cv2.dilate(
            mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )

        edge_masks = {}
        if self.edge_type in {"outer", "both"}:
            edge_masks["outer"] = dilated_mask - mask.astype(np.uint8)
        if self.edge_type in {"inner", "both"}:
            edge_masks["inner"] = mask.astype(np.uint8) - eroded_mask

        return edge_masks

    def transform(self, results: dict) -> dict:
        mask_arrays = results["gt_masks"].masks
        img_shape = results["img"].shape[:2]

        if mask_arrays.size == 0:
            if self.edge_type in {"inner", "both"}:
                # edge_masks_inner: (N, H, W)
                results["edge_masks_inner"] = np.empty((0, *img_shape), dtype=np.uint8)
            if self.edge_type in {"outer", "both"}:
                # edge_masks_outer: (N, H, W)
                results["edge_masks_outer"] = np.empty((0, *img_shape), dtype=np.uint8)
            return results

        edge_data = [self.compute_edge_mask(mask) for mask in mask_arrays]

        if self.edge_type in {"inner", "both"}:
            # edge_masks_inner: (N, H, W)
            results["edge_masks_inner"] = np.stack([e["inner"] for e in edge_data])
        if self.edge_type in {"outer", "both"}:
            # edge_masks_outer: (N, H, W)
            results["edge_masks_outer"] = np.stack([e["outer"] for e in edge_data])
        if self.edge_type == "both":
            results["edge_masks_combined"] = (
                results["edge_masks_inner"].astype(bool)
                | results["edge_masks_outer"].astype(bool)
            ).astype(
                np.uint8
            )  # (N, H, W)

        return results


@TRANSFORMS.register_module()
class SampleSkeletonPoints(BaseTransform):
    def __init__(
        self,
        num_points: int = 30,
        skeleton_key: str = "skeleton_masks",
        test: bool = False,
    ):
        self.num_points = num_points
        self.skeleton_key = skeleton_key
        self.test = test

    @cache_randomness
    def random_indices(self, n: int, k: int) -> np.ndarray:
        if n <= k:
            return np.arange(n)
        return np.random.choice(n, k, replace=False)

    def transform(self, results: dict) -> dict:
        skeleton_masks = results[self.skeleton_key]  # (N, H, W)
        N = skeleton_masks.shape[0]
        K = self.num_points
        sampled = np.zeros((N, K, 2), dtype=np.float32)

        for i, mask in enumerate(skeleton_masks):
            pts = np.column_stack(np.nonzero(mask))[:, ::-1]  # (x, y)
            if len(pts) > 0:
                idxs = self.random_indices(len(pts), K)
                selected = pts[idxs].astype(np.float32)
                sampled[i, : len(selected)] = selected

        if self.test:
            scale_factor = results["scale_factor"]
            sampled[..., 0] = sampled[..., 0] * scale_factor[0] + 0.5
            sampled[..., 1] = sampled[..., 1] * scale_factor[1] + 0.5
        else:
            sampled += 0.5

        results["skeleton_sampled_points"] = sampled  # (N, K, 2)
        return results


@TRANSFORMS.register_module()
class GetSkeletonMask(BaseTransform):
    def __init__(self):
        pass

    def compute_skeleton(self, mask: np.ndarray) -> np.ndarray:
        return skeletonize(mask.astype(bool)).astype(np.uint8)

    def transform(self, results: dict) -> dict:
        mask_arrays = results["gt_masks"].masks  # list of (H, W) arrays
        img_shape = results["img"].shape[:2]

        if mask_arrays.size == 0:
            results["skeleton_masks"] = np.empty((0, *img_shape), dtype=np.uint8)
            return results

        skeletons = [self.compute_skeleton(mask) for mask in mask_arrays]
        results["skeleton_masks"] = np.stack(skeletons)
        return results


@TRANSFORMS.register_module()
class GetAnatomicalPoles(BaseTransform):
    def __init__(
        self,
        edge_key: str = "edge_masks_inner",
        top_x: Union[int, float] = 0.040,
        pole_erosion: int = 10,
        w_main: float = 0.6,
        w_ortho: float = 0.4,
        test: bool = False,
    ):
        self.type = PromptType.MAJMIN
        self.edge_key = edge_key
        self.top_x = top_x  # can be absolute or % of available points
        self.pole_erosion = pole_erosion
        self.w_main = w_main
        self.w_ortho = w_ortho
        self.test = test  # rescale point coordinate as img

    def compute_top_x(self, n_pts: int) -> int:
        if isinstance(self.top_x, float):
            return max(1, int(n_pts * self.top_x))
        return self.top_x

    def pick_pool(
        self,
        proj: np.ndarray,
        axis_idx: int,
        kind: Literal["min", "max"],
        top_k: int,
        w_main: float = 0.6,
        w_ortho: float = 0.4,
    ) -> np.ndarray:
        """
        Select top_k points that have extreme projection on axis_idx
        and minimal projection on the orthogonal axis.

        Args:
            proj: (N, 2) projected points.
            axis_idx: 0 for major axis, 1 for minor axis.
            kind: 'min' or 'max' projection on main axis.
            top_k: number of points to return.
            w_main: weight for main axis score.
            w_ortho: weight for orthogonal axis penalty.

        Returns:
            top_k indices in the original array.
        """
        main = proj[:, axis_idx]
        ortho = np.abs(proj[:, 1 - axis_idx])

        # Normalise
        main_norm = (main - main.min()) / (main.max() - main.min() + 1e-8)
        ortho_norm = ortho / (ortho.max() + 1e-8)

        # for main axis, use only proj, not ortho
        # if axis_idx == 0:
        #     w_main = 1
        #     w_ortho = 0
        if kind == "min":
            score = -w_main * main_norm + w_ortho * ortho_norm
        else:
            score = w_main * main_norm + w_ortho * ortho_norm

        top_idxs = np.argsort(score)[:top_k]
        return top_idxs

    def get_projection(
        self, pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute center, axes and projection using PCA."""
        pca = PCA(2).fit(pts)
        dirs = pca.components_
        center = pca.mean_
        proj = (pts - center) @ dirs.T
        return center, dirs, proj

    @cache_randomness
    def random_index(self, n: int) -> int:
        return np.random.randint(n)

    def transform(self, results: dict) -> dict:
        # breakpoint()  # see if we have box
        edge_masks = results.get(self.edge_key)
        pools = []  # List[(min0, max0, min1, max1)] per instance
        dilated_pool_masks = []

        centers = []
        axes = []

        for i in range(edge_masks.shape[0]):
            boundary_mask = edge_masks[i]
            pts = np.column_stack(np.nonzero(boundary_mask))[:, ::-1]  # (x, y)

            center, dirs, proj = self.get_projection(pts)
            centers.append(center)
            axes.append(dirs)

            # dirs = np.array([[1, 0], [0, 1]])  # x and y axes
            # center = pts.mean(axis=0)
            # centers.append(center)
            # axes.append(dirs)
            # proj = (pts - center) @ dirs.T

            n_fg = np.count_nonzero(boundary_mask)
            n_points = (
                max(1, int(self.top_x * n_fg))
                if isinstance(self.top_x, float)
                else self.top_x
            )

            # pool_min0 = self.pick_pool(proj[:, 0], 'min', n_points)
            # pool_max0 = self.pick_pool(proj[:, 0], 'max', n_points)
            # pool_min1 = self.pick_pool(proj[:, 1], 'min', n_points)
            # pool_max1 = self.pick_pool(proj[:, 1], 'max', n_points)

            pool_min0 = self.pick_pool(
                proj, 0, "min", n_points, w_main=self.w_main, w_ortho=self.w_ortho
            )
            pool_max0 = self.pick_pool(
                proj, 0, "max", n_points, w_main=self.w_main, w_ortho=self.w_ortho
            )
            pool_min1 = self.pick_pool(
                proj, 1, "min", n_points, w_main=self.w_main, w_ortho=self.w_ortho
            )
            pool_max1 = self.pick_pool(
                proj, 1, "max", n_points, w_main=self.w_main, w_ortho=self.w_ortho
            )

            pools_i = [
                pts[pool_min0],
                pts[pool_max0],
                pts[pool_min1],
                pts[pool_max1],
            ]  # Liste de 4 arrays

            dilated_pool_masks_i = []
            H, W = boundary_mask.shape
            kernel = np.ones((self.pole_erosion, self.pole_erosion), np.uint8)
            for j in range(4):
                mask = np.zeros((H, W), dtype=np.uint8)
                for x, y in pools_i[j]:
                    mask[int(round(y)), int(round(x))] = 1
                dilated = cv2.dilate(mask, kernel)
                dilated_pool_masks_i.append(dilated)
                yx = np.column_stack(np.nonzero(dilated))
                xy = yx[:, ::-1]
                pools_i[j] = xy

            pools.append(pools_i)
            dilated_pool_masks.append(dilated_pool_masks_i)

        results["pca_centers"] = np.array(centers)  # shape (N, 2)
        results["pca_axes"] = np.array(axes)  # shape (N, 2, 2)

        results["anatomical_pole_area"] = dilated_pool_masks
        scale_factor = results["scale_factor"]

        if pools:
            final_pools = []

            for pools_i in pools:
                pooled = []
                for region in pools_i:
                    if len(region) == 0:
                        pooled.append(
                            np.array([0, 0])
                        )  # ou skip/error selon ton besoin
                    else:
                        # idx = np.random.randint(len(region))
                        idx = self.random_index(len(region))
                        pooled.append(region[idx])  # shape (2,)

                final_pools.append(pooled)  # 4 points (x, y) pour une instance

            results["anatomical_pole_pools"] = np.array(final_pools)  # shape (N, 4, 2)
            results["prompt_types"] = results["prompt_types"] = np.full(
                (len(final_pools),), self.type.value
            )

            if self.test:
                # Scale the points using the scale factor
                # and move to pixel center
                results["anatomical_pole_pools"][..., 0] = (
                    results["anatomical_pole_pools"][..., 0] * scale_factor[0]
                ) + 0.5
                results["anatomical_pole_pools"][..., 1] = (
                    results["anatomical_pole_pools"][..., 1] * scale_factor[1]
                ) + 0.5
            else:
                results["anatomical_pole_pools"][..., 0] = (
                    results["anatomical_pole_pools"][..., 0] + 0.5
                )
                results["anatomical_pole_pools"][..., 1] = (
                    results["anatomical_pole_pools"][..., 1] + 0.5
                )
        else:
            results["anatomical_pole_pools"] = np.empty((0, 4, 2))
            results["prompt_type"] = np.empty((0))

        return results


@TRANSFORMS.register_module()
class GetAnatomicalPolesFixedAxis(GetAnatomicalPoles):
    """fixed axis for extreme points"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = PromptType.EXTREME

    def get_projection(
        self, pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = pts.mean(axis=0)
        dirs = np.array([[1, 0], [0, 1]])  # fixed x/y axes
        proj = (pts - center) @ dirs.T
        return center, dirs, proj


@TRANSFORMS.register_module()
class GetSkeletonAnatomicalPoles(GetAnatomicalPoles):
    def get_projection(
        self, pts: np.ndarray, pts_to_proj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute center, axes and projection using PCA."""
        pca = PCA(2).fit(pts)
        dirs = pca.components_
        center = pca.mean_
        proj = (pts_to_proj - center) @ dirs.T
        return center, dirs, proj

    def transform(self, results: dict) -> dict:
        # breakpoint()  # see if we have box
        edge_masks = results.get(self.edge_key)
        skeleton_masks = results.get("skeleton_masks")
        pools = []  # List[(min0, max0, min1, max1)] per instance
        dilated_pool_masks = []

        centers = []
        axes = []

        for i in range(edge_masks.shape[0]):
            boundary_mask = edge_masks[i]
            skeleton_mask = skeleton_masks[i]
            pts = np.column_stack(np.nonzero(boundary_mask))[:, ::-1]  # (x, y)
            pts_skeleton = np.column_stack(np.nonzero(skeleton_mask))[:, ::-1]  # (x, y)

            # difference here: projection done on skeleton points

            center, dirs, proj = self.get_projection(pts_skeleton, pts)
            centers.append(center)
            axes.append(dirs)

            n_fg = np.count_nonzero(boundary_mask)
            n_points = (
                max(1, int(self.top_x * n_fg))
                if isinstance(self.top_x, float)
                else self.top_x
            )

            pool_min0 = self.pick_pool(
                proj, 0, "min", n_points, w_main=self.w_main, w_ortho=self.w_ortho
            )
            pool_max0 = self.pick_pool(
                proj, 0, "max", n_points, w_main=self.w_main, w_ortho=self.w_ortho
            )
            pool_min1 = self.pick_pool(
                proj, 1, "min", n_points, w_main=self.w_main, w_ortho=self.w_ortho
            )
            pool_max1 = self.pick_pool(
                proj, 1, "max", n_points, w_main=self.w_main, w_ortho=self.w_ortho
            )

            pools_i = [
                pts[pool_min0],
                pts[pool_max0],
                pts[pool_min1],
                pts[pool_max1],
            ]  # Liste de 4 arrays

            dilated_pool_masks_i = []
            H, W = boundary_mask.shape
            kernel = np.ones((self.pole_erosion, self.pole_erosion), np.uint8)
            for j in range(4):
                mask = np.zeros((H, W), dtype=np.uint8)
                for x, y in pools_i[j]:
                    mask[int(round(y)), int(round(x))] = 1
                dilated = cv2.dilate(mask, kernel)
                dilated_pool_masks_i.append(dilated)
                yx = np.column_stack(np.nonzero(dilated))
                xy = yx[:, ::-1]
                pools_i[j] = xy

            pools.append(pools_i)
            dilated_pool_masks.append(dilated_pool_masks_i)

        results["pca_centers"] = np.array(centers)  # shape (N, 2)
        results["pca_axes"] = np.array(axes)  # shape (N, 2, 2)
        results["anatomical_pole_area"] = dilated_pool_masks
        scale_factor = results["scale_factor"]

        if pools:
            final_pools = []

            for pools_i in pools:
                pooled = []
                for region in pools_i:
                    if len(region) == 0:
                        pooled.append(
                            np.array([0, 0])
                        )  # ou skip/error selon ton besoin
                    else:
                        # idx = np.random.randint(len(region))
                        idx = self.random_index(len(region))
                        pooled.append(region[idx])  # shape (2,)

                final_pools.append(pooled)  # 4 points (x, y) pour une instance

            results["anatomical_pole_pools"] = np.array(final_pools)  # shape (N, 4, 2)
            results["prompt_types"] = results["prompt_types"] = np.full(
                (len(final_pools),), self.type.value
            )

            if self.test:
                # Scale the points using the scale factor
                # and move to pixel center
                results["anatomical_pole_pools"][..., 0] = (
                    results["anatomical_pole_pools"][..., 0] * scale_factor[0]
                ) + 0.5
                results["anatomical_pole_pools"][..., 1] = (
                    results["anatomical_pole_pools"][..., 1] * scale_factor[1]
                ) + 0.5
            else:
                results["anatomical_pole_pools"][..., 0] = (
                    results["anatomical_pole_pools"][..., 0] + 0.5
                )
                results["anatomical_pole_pools"][..., 1] = (
                    results["anatomical_pole_pools"][..., 1] + 0.5
                )
        else:
            results["anatomical_pole_pools"] = np.empty((0, 4, 2))
            results["prompt_type"] = np.empty((0))

        return results


@TRANSFORMS.register_module()
class GetCornerAnatomicalPoles(GetAnatomicalPoles):
    def pick_pool(
        self,
        proj: np.ndarray,
        kind: Literal["top-left", "top-right", "bottom-left", "bottom-right"],
        top_k: int,
        w_main: float = 0.5,
        w_ortho: float = 0.5,
    ) -> np.ndarray:
        """
        Select top_k points closest to a PCA-defined corner.

        Args:
            proj: (N, 2) projected points (PCA space).
            kind: which corner to select.
            top_k: number of points to return.
            w_main: weight for first PCA axis.
            w_ortho: weight for second PCA axis.

        Returns:
            top_k indices of selected points.
        """
        x, y = proj[:, 0], proj[:, 1]

        # Normalize
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)

        if kind == "top-left":
            score = -w_main * x_norm - w_ortho * y_norm
        elif kind == "top-right":
            score = w_main * x_norm - w_ortho * y_norm
        elif kind == "bottom-left":
            score = -w_main * x_norm + w_ortho * y_norm
        elif kind == "bottom-right":
            score = w_main * x_norm + w_ortho * y_norm
        else:
            raise ValueError("Invalid kind")

        return np.argsort(score)[:top_k]

    def transform(self, results: dict) -> dict:
        # breakpoint()  # see if we have box
        edge_masks = results.get(self.edge_key)
        skeleton_masks = results.get("skeleton_masks")
        pools = []  # List[(min0, max0, min1, max1)] per instance
        dilated_pool_masks = []

        centers = []
        axes = []

        for i in range(edge_masks.shape[0]):
            boundary_mask = edge_masks[i]
            pts = np.column_stack(np.nonzero(boundary_mask))[:, ::-1]  # (x, y)

            center, dirs, proj = self.get_projection(pts)
            centers.append(center)
            axes.append(dirs)

            n_fg = np.count_nonzero(boundary_mask)
            n_points = (
                max(1, int(self.top_x * n_fg))
                if isinstance(self.top_x, float)
                else self.top_x
            )

            # difference here: use top left, top right, bottom left, bottom right
            pool_min0 = self.pick_pool(proj, "top-left", n_points)
            pool_max1 = self.pick_pool(proj, "bottom-right", n_points)
            pool_max0 = self.pick_pool(proj, "top-right", n_points)
            pool_min1 = self.pick_pool(proj, "bottom-left", n_points)

            pools_i = [
                pts[pool_min0],
                pts[pool_max0],
                pts[pool_min1],
                pts[pool_max1],
            ]  # Liste de 4 arrays

            dilated_pool_masks_i = []
            H, W = boundary_mask.shape
            kernel = np.ones((self.pole_erosion, self.pole_erosion), np.uint8)
            for j in range(4):
                mask = np.zeros((H, W), dtype=np.uint8)
                for x, y in pools_i[j]:
                    mask[int(round(y)), int(round(x))] = 1
                dilated = cv2.dilate(mask, kernel)
                dilated_pool_masks_i.append(dilated)
                yx = np.column_stack(np.nonzero(dilated))
                xy = yx[:, ::-1]
                pools_i[j] = xy

            pools.append(pools_i)
            dilated_pool_masks.append(dilated_pool_masks_i)

        results["pca_centers"] = np.array(centers)  # shape (N, 2)
        results["pca_axes"] = np.array(axes)  # shape (N, 2, 2)
        results["anatomical_pole_area"] = dilated_pool_masks
        scale_factor = results["scale_factor"]

        if pools:
            final_pools = []

            for pools_i in pools:
                pooled = []
                for region in pools_i:
                    if len(region) == 0:
                        pooled.append(
                            np.array([0, 0])
                        )  # ou skip/error selon ton besoin
                    else:
                        # idx = np.random.randint(len(region))
                        idx = self.random_index(len(region))
                        pooled.append(region[idx])  # shape (2,)

                final_pools.append(pooled)  # 4 points (x, y) pour une instance

            results["anatomical_pole_pools"] = np.array(final_pools)  # shape (N, 4, 2)

            if self.test:
                # Scale the points using the scale factor
                # and move to pixel center
                results["anatomical_pole_pools"][..., 0] = (
                    results["anatomical_pole_pools"][..., 0] * scale_factor[0]
                ) + 0.5
                results["anatomical_pole_pools"][..., 1] = (
                    results["anatomical_pole_pools"][..., 1] * scale_factor[1]
                ) + 0.5
            else:
                results["anatomical_pole_pools"][..., 0] = (
                    results["anatomical_pole_pools"][..., 0] + 0.5
                )
                results["anatomical_pole_pools"][..., 1] = (
                    results["anatomical_pole_pools"][..., 1] + 0.5
                )
        else:
            results["anatomical_pole_pools"] = np.empty((0, 4, 2))

        return results


@TRANSFORMS.register_module()
class GetSinglePointFromMask(BaseTransform):
    def __init__(self, test=False, mixed_format: bool = False):
        self.test = test
        self.type = PromptType.POINT
        self.mixed_format = mixed_format

    def _get_point(self, results):
        mask_arrays = results["gt_masks"].masks
        x_scale, y_scale = results.get("scale_factor", (1.0, 1.0))
        points_list = []

        for mask in mask_arrays:
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                x, y = 0.0, 0.0
            else:
                idx = np.random.randint(len(xs))
                x, y = xs[idx], ys[idx]  # x = col, y = row

            if self.test:
                x = x * x_scale + 0.5
                y = y * y_scale + 0.5
            else:
                x = x + 0.5
                y = y + 0.5

            points_list.append(np.array([[x, y]], dtype=np.float32))

        return points_list

    def transform(self, results):
        points = self._get_point(results)
        num_masks = len(points)

        if self.mixed_format:
            # num_masks, 4, 2   We use 4 to "pad" as its used in mixed prompt encoder (4 max for 4 extreme), use 1 if you plan to not need to pad
            results["anatomical_pole_pools"] = np.zeros(
                (num_masks, 4, 2), dtype=np.float32
            )
            results["prompt_types"] = np.full((num_masks,), self.type.value, dtype=int)

            for i in range(num_masks):
                results["anatomical_pole_pools"][i, 0] = points[i]
        else:
            results["points"] = np.zeros((num_masks, 1, 2), dtype=np.float32)
            results["prompt_types"] = np.full((num_masks,), self.type.value, dtype=int)

            for i in range(num_masks):
                results["points"][i, 0] = points[i]

        return results


@TRANSFORMS.register_module()
class GetPointBox(BaseTransform):
    def __init__(
        self, normalize: bool = False, max_jitter: float = 0.05, test: bool = False
    ):
        """
        Initializes the GetPointBox object.

        Args:
            max_jitter (float): Max jitter to move each box corner as a fraction of box dimensions.
            normalize (bool): Whether to normalize the coordinates of the point.
            test (bool): Whether the class is in test mode.
        """
        self.normalize = normalize
        self.max_jitter = max_jitter
        self.test = test
        self.type = PromptType.BOX

    def _apply_jitter(
        self,
        x_points: torch.Tensor,
        y_points: torch.Tensor,
        box_width: torch.Tensor,
        box_height: torch.Tensor,
    ):
        """
        Apply jitter to the box corners.

        Args:
            x_points (torch.Tensor): Tensor containing xmin and xmax.
            y_points (torch.Tensor): Tensor containing ymin and ymax.
            box_width (torch.Tensor): Width of the bounding box.
            box_height (torch.Tensor): Height of the bounding box.

        Returns:
            (torch.Tensor, torch.Tensor): Jittered x_points and y_points.
        """
        # Generate different jitter for each corner
        jitter_x = (
            torch.rand(2) * 2 * self.max_jitter - self.max_jitter
        )  # Uniform random jitter between -max_jitter and +max_jitter
        jitter_y = torch.rand(2) * 2 * self.max_jitter - self.max_jitter

        # Apply jitter to each corner scaled by box width/height
        x_points += jitter_x * box_width
        y_points += jitter_y * box_height

        return x_points, y_points

    def _getPointBox(self, results) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        bbox_arrays = results["gt_bboxes"].tensor
        img_key = "img_shape"
        if self.normalize:
            img_height, img_width = results[img_key]
        else:
            img_height, img_width = 1, 1
        if self.test:
            x_scale, y_scale = results["scale_factor"]

        points_list = []

        for bbox in bbox_arrays:
            xmin, ymin, xmax, ymax = bbox
            x_points = torch.tensor([xmin, xmax], dtype=bbox.dtype)
            y_points = torch.tensor([ymin, ymax], dtype=bbox.dtype)

            box_width = xmax - xmin
            box_height = ymax - ymin

            # Apply jitter to the corners
            x_points, y_points = self._apply_jitter(
                x_points, y_points, box_width, box_height
            )

            if self.test:
                x_points = x_points * x_scale / img_width + 0.5
                y_points = y_points * y_scale / img_height + 0.5
            else:
                x_points = x_points / img_width + 0.5
                y_points = y_points / img_height + 0.5

            index_candidat = torch.stack((x_points, y_points), dim=-1)
            points_list.append(index_candidat)

        return points_list

    def transform(self, results):
        boxes = self._getPointBox(results)
        N = len(boxes)

        # pre-allocate zeros
        results["anatomical_pole_pools"] = torch.zeros((N, 4, 2), dtype=torch.float32)
        results["prompt_types"] = np.full((N,), self.type.value, dtype=int)

        # fill with real boxes if available
        for i, b in enumerate(boxes):
            results["anatomical_pole_pools"][i][:2] = b

        return results
