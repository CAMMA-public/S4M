from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData, LabelData
from mmengine.visualization import Visualizer

from mmdet.evaluation import INSTANCE_OFFSET
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from mmdet.visualization.palette import _get_adaptive_scales, get_palette, jitter_color
from mmdet.visualization import DetLocalVisualizer

# set to -1 for no instance filtering
INSTANCE_ID = -1
POINT_SCALING = 6


@VISUALIZERS.register_module()
class ExtremeVisualizer(DetLocalVisualizer):
    def fade_background(
        self,
        image: np.ndarray,
        image_faded: np.ndarray,
        instances: ["InstanceData"],
    ) -> np.ndarray:
        if "masks" not in instances:
            return self.get_image()

        masks = instances.masks
        if isinstance(masks, torch.Tensor):
            masks = masks.numpy()
        elif isinstance(masks, (PolygonMasks, BitmapMasks)):
            masks = masks.to_ndarray()

        masks = masks.astype(bool)
        if masks.shape == (0,):
            return self.get_image()
        for mask in masks:
            image_faded[mask] = image[mask]
        return image_faded

    def extract_instances_with_transparency(
        self,
        image: np.ndarray,
        instances: ["InstanceData"],
        anatomical_pole_pools,
        with_pool=True,
    ):
        if "masks" not in instances:
            return []

        masks = instances.masks
        if isinstance(masks, torch.Tensor):
            masks = masks.numpy()
        elif isinstance(masks, (PolygonMasks, BitmapMasks)):
            masks = masks.to_ndarray()

        masks = masks.astype(bool)
        if masks.shape == (0,):
            return []

        result = []
        for i, mask in enumerate(masks):
            if with_pool and i < len(anatomical_pole_pools):
                # Combine mask with the union of its 4 extreme pole masks
                pool_mask = np.logical_or.reduce(anatomical_pole_pools[i])
                mask = np.logical_or(mask, pool_mask)
            rgba = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3][mask] = image[mask]
            rgba[..., 3][mask] = 255
            result.append(rgba)
        return result

    def plot_poles(self, gt_img_data: np.ndarray, anatomical_pole_pools):
        self.set_image(gt_img_data)
        polygons = []
        colors_extended = []
        for i_idx, instance_idx in enumerate(anatomical_pole_pools):
            if INSTANCE_ID >= 0:
                if i_idx != INSTANCE_ID:
                    continue
            for i, (color, pool) in enumerate(
                zip(["magenta", "cyan", "#00AA80", "darkorange"], instance_idx)
            ):  # Iterate over the second dimension (4 extreme points)
                contours, _ = bitmap_to_polygon(pool)
                polygons.extend(contours)
                colors_extended.extend([color] * len(contours))

        self.draw_polygons(polygons, edge_colors=colors_extended, alpha=0.9)

        return self.get_image()

    def _draw_area(self, image: np.ndarray, instances: ["InstanceData"]) -> np.ndarray:
        if "masks" in instances:
            masks = instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            masks = masks.astype(bool)
            if masks.shape == (0,):
                return self.get_image()

    def _draw_instances(
        self,
        image: np.ndarray,
        instances: ["InstanceData"],
        meta_instances,
        classes: Optional[List[str]],
        palette: Optional[List[tuple]],
        draw_majmin_lines: bool = False,
        draw_PCA_basis: bool = False,
        draw_points: bool = True,
        draw_box: bool = False,
        draw_intermediate: int = -1,
    ) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if "masks" in instances:
            labels = instances.labels
            masks = instances.masks

            # (Pdb) instances.intermediate_masks.shape
            # torch.Size([1, 3, 533, 958])
            if "interactive_points" in instances and draw_intermediate >= 0:
                masks = instances.intermediate_masks[:, draw_intermediate]

            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            masks = masks.astype(bool)
            if masks.shape == (0,):
                return self.get_image()

            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            # colors = [jitter_color(mask_palette[label]) for label in labels]
            colors = [mask_palette[label] for label in labels]
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            polygons = []
            colors_extended = []
            if INSTANCE_ID >= 0:
                masks = masks[INSTANCE_ID][None]
                colors = [colors[INSTANCE_ID]]
            for mask, color in zip(masks, colors):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
                colors_extended.extend([color] * len(contours))

            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)
            self.draw_polygons(polygons, edge_colors=colors_extended, alpha=0.9)

            if draw_box and "bboxes" in instances and instances.bboxes.sum() > 0:
                bboxes = instances.bboxes
                labels = instances.labels

                self.draw_bboxes(
                    bboxes,
                    edge_colors=colors,
                    alpha=self.alpha,
                    line_widths=self.line_width,
                )

        if "anatomical_pole_pools" in instances and instances.bboxes.sum() > 0:
            pts = instances.anatomical_pole_pools

            # Liaisons entre magenta (i=0) et cyan (i=1)
            majA_points = pts[:, 0, :]
            majB_points = pts[:, 1, :]

            if INSTANCE_ID >= 0:
                majA_points = pts[INSTANCE_ID, 0, :][None]
                majB_points = pts[INSTANCE_ID, 1, :][None]

            majx_datas = np.stack([majA_points[:, 0], majB_points[:, 0]], axis=1)
            majy_datas = np.stack([majA_points[:, 1], majB_points[:, 1]], axis=1)

            minA_points = pts[:, 2, :]
            minB_points = pts[:, 3, :]

            if INSTANCE_ID >= 0:
                minA_points = pts[INSTANCE_ID, 2, :][None]
                minB_points = pts[INSTANCE_ID, 3, :][None]

            minx_datas = np.stack([minA_points[:, 0], minB_points[:, 0]], axis=1)
            miny_datas = np.stack([minA_points[:, 1], minB_points[:, 1]], axis=1)

            if draw_majmin_lines:
                self.draw_lines(
                    minx_datas, miny_datas, colors="w", line_styles="--", line_widths=2
                )

                self.draw_lines(
                    majx_datas,
                    majy_datas,
                    colors="azure",
                    line_styles="-",
                    line_widths=2,
                )

            # TODO draw link between each pair (magenta and cyan of the same instance)
            offsets = {
                "→": torch.tensor([0.0, 1.3]),
                "←": torch.tensor([0.0, 1.3]),
                "↑": torch.tensor([0.0, 2.0]),
                "↓": torch.tensor([0.0, 2.0]),
                # "M": torch.tensor([0.0, 1.3]),
                "M": torch.tensor([0.0, 1.3]),
                "m": torch.tensor([0.0, 0.0]),
                # "m": torch.tensor([0.0, 0.0]),
            }

            if draw_points:
                # compute center of the 4 points (mean across axis=1)
                centers = pts.mean(dim=1, keepdim=True).to(
                    pts.device
                )  # shape (N, 1, 2)

                for i, (arrow, color) in enumerate(
                    # zip(["→", "←", "↓", "↑"], ["magenta", "cyan", "#00AA80", "darkorange"])
                    zip(
                        ["M", "M", "m", "m"],
                        # ["→", "←", "↓", "↑"],
                        # [">", "<", "^", "v"],
                        ["#1f77b4", "#1f77b4", "#aec7e8", "#aec7e8"],
                        # ["magenta", "cyan", "#00AA80", "darkorange"],
                    )
                ):  # Iterate over the second dimension (4 extreme points)
                    points = pts[
                        :, i, :
                    ]  # Extract points for all instances at index `i`
                    shift = offsets[arrow].to(points.device).unsqueeze(0)  # shape (1,2)
                    pos_shifted = points + shift  # keep (1,2)
                    if INSTANCE_ID >= 0:
                        points = pts[INSTANCE_ID, i, :][None]

                    # draw small white line from point to center
                    center = centers[:, 0, :]
                    # --- draw white line from each point to its center ---
                    x_datas = np.stack(
                        [points[:, 0].cpu().numpy(), center[:, 0].cpu().numpy()], axis=1
                    )
                    y_datas = np.stack(
                        [points[:, 1].cpu().numpy(), center[:, 1].cpu().numpy()], axis=1
                    )
                    self.draw_lines(
                        x_datas=x_datas,
                        y_datas=y_datas,
                        colors="white",
                        line_styles="-",
                        line_widths=1.5,
                    )

                    self.draw_points(
                        points,
                        colors="w",
                        sizes=np.array([[40 * POINT_SCALING]], dtype=np.float32),
                    )
                    self.draw_points(
                        points,
                        # colors=[colors[i]],
                        colors=color,
                        sizes=np.array([[25 * POINT_SCALING]], dtype=np.float32),
                    )

                    texts = [arrow] * len(points)
                    self.draw_texts(
                        texts=texts,
                        positions=pos_shifted,  # np.ndarray or torch.Tensor of shape (K, 2)
                        font_sizes=14,  # ou [20]*K
                        colors="navy",
                        vertical_alignments="center",
                        horizontal_alignments="center",
                        font_families="sans-serif",
                    )
                    # self.draw_texts(
                    #     texts=texts,
                    #     positions=points,  # np.ndarray or torch.Tensor of shape (K, 2)
                    #     font_sizes=9,  # ou [20]*K
                    #     colors="",
                    #     vertical_alignments="center",
                    #     horizontal_alignments="center",
                    #     font_families="sans-serif",
                    # )

            if (
                True
                and "interactive_points" in instances
                and instances.bboxes.sum() > 0
            ):
                # torch.Size([n_inst, n_points, 2])
                interactive_points = instances.interactive_points

                # (Pdb) instances.points.shape
                # torch.Size([n_inst, 1, 2])

                # TODO select only the first draw_intermediate number of points
                # if draw_intermediate >= 0:

                # torch.Size([n_inst, n_points, 1])
                interactive_points_types = instances.interactive_points_types.squeeze(
                    -1
                )  # shape: [n_inst, n_points]

                if draw_intermediate > 0:
                    interactive_points = interactive_points[:, :draw_intermediate]
                    interactive_points_types = interactive_points_types[
                        :, :draw_intermediate
                    ]

                if draw_intermediate != 0:
                    # Couleurs par type, ex: 0 = rouge, 1 = vert, 2 = bleu
                    type_colors = {0: "blue", 1: "red", 2: "lime"}
                    colors = [
                        type_colors.get(pt_type.item(), "white")
                        for inst_types in interactive_points_types
                        for pt_type in inst_types
                    ]

                    points_flat = interactive_points.view(
                        -1, 2
                    )  # [n_inst * n_points, 2]
                    self.draw_points(
                        points_flat,
                        colors="w",
                        sizes=np.array([[40 * POINT_SCALING]], dtype=np.float32),
                    )
                    self.draw_points(
                        points_flat,
                        colors=colors,
                        sizes=np.array(
                            [[25 * POINT_SCALING]] * len(colors), dtype=np.float32
                        ),
                    )

                # initial point
                # self.draw_points(
                #     instances.points.view(-1, 2),
                #     colors="w",
                #     marker="*",
                #     sizes=np.array([[40 * (7 + POINT_SCALING)]], dtype=np.float32),
                # )
                # self.draw_points(
                #     instances.points.view(-1, 2),
                #     colors="blue",
                #     marker="*",
                #     sizes=np.array([[25 * (4 + POINT_SCALING)]], dtype=np.float32),
                # )

            # (Pdb) meta_instances.candidates[0][0].shape
            # (811, 2)
            # (Pdb) meta_instances.candidates[0][1].shape
            # (773, 2)

            # if draw_candidates:
            #     for candidate in meta_instances.candidates:
            #         for side in candidate:
            #             self.draw_points(
            #                 side,
            #                 colors="red",
            #                 sizes=np.array([[1]], dtype=np.float32),
            #             )

            # plot the PCA axis
            # if draw_PCA_basis and "pca_centers" in meta_instances.keys():
            if draw_PCA_basis:
                for idx, (center, axes) in enumerate(
                    zip(
                        meta_instances.pca_centers,
                        meta_instances.pca_axes,
                    )
                ):
                    if INSTANCE_ID >= 0:
                        if idx != INSTANCE_ID:
                            continue
                    for vec, c in zip(axes, ["#1f77b4", "#aec7e8"]):
                        pt1 = center
                        pt2 = center + vec * 55
                        x_data = np.array([[pt1[0], pt2[0]]])
                        y_data = np.array([[pt1[1], pt2[1]]])
                        self.draw_lines(x_data, y_data, colors=c, line_widths=3)

                # # draw skeleton
                # kernel = np.ones((5, 5), np.uint8)
                # dilated = np.stack([
                #     cv2.dilate(mask, kernel, iterations=1) for mask in meta_instances.skeleton_masks
                # ])
                # self.draw_binary_masks(dilated.astype(bool), colors='w', alphas=1.0)

        return self.get_image()

    @master_only
    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: Optional["DetDataSample"] = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = False,
        wait_time: float = 0,
        # TODO: Supported in mmengine's Viusalizer.
        out_file: Optional[str] = None,
        pred_score_thr: float = 0.3,
        step: int = 0,
    ) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)

        classes = self.dataset_meta.get("classes", None)[1:]
        palette = self.dataset_meta.get("palette", None)[1:]

        alpha = 0.6  # Transparency level (0 = fully transparent, 1 = original color)
        background = np.full_like(image, 0)  # White background

        # Blend the image with the background
        image_faded = cv2.addWeighted(image, alpha, background, 1 - alpha, 0)
        image_faded = self.fade_background(image, image_faded, data_sample.gt_instances)

        gt_img_data = None
        raw_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            raw_img_data = image
            gt_img_data = image
            canvas_img_data = image
            pred_img_data = image
            if "gt_instances" in data_sample:
                # rescale input points to original size
                if data_sample.gt_instances.bboxes.sum() > 0:

                    # scale_y, scale_x = data_sample.scale_factor  # Unpack scale factors
                    scale_x, scale_y = data_sample.scale_factor
                    # Apply scaling to the extreme points
                    data_sample.gt_instances.anatomical_pole_pools = data_sample.gt_instances.anatomical_pole_pools / torch.tensor(
                        [scale_x, scale_y],
                        device=data_sample.gt_instances.anatomical_pole_pools.device,
                    )

                    if "interactive_points" in data_sample.gt_instances:
                        data_sample.gt_instances.interactive_points = data_sample.gt_instances.interactive_points / torch.tensor(
                            [scale_x, scale_y],
                            device=data_sample.gt_instances.interactive_points.device,
                        )
                    if "points" in data_sample.gt_instances:
                        data_sample.gt_instances.points = (
                            data_sample.gt_instances.points
                            / torch.tensor(
                                [scale_x, scale_y],
                                device=data_sample.gt_instances.points.device,
                            )
                        )

                # gt_img_data = self._draw_instances(image_faded.copy(),
                #                                    data_sample.gt_instances,
                #                                    data_sample,
                #                                    classes, palette,
                #                                    draw_majmin_lines=True,
                #                                    draw_PCA_basis=True)
                gt_img_data = self._draw_instances(
                    image_faded.copy(),
                    data_sample.gt_instances,
                    data_sample,
                    classes,
                    palette,
                    draw_majmin_lines=False,
                    draw_PCA_basis=False,
                    draw_points=True,
                    draw_intermediate=-1,
                )

                gt_img_data_axis = self._draw_instances(
                    image_faded.copy(),
                    data_sample.gt_instances,
                    data_sample,
                    classes,
                    palette,
                    draw_majmin_lines=False,
                    draw_PCA_basis=True,
                    draw_points=False,
                )

                # gt_img_data_axis_point = self._draw_instances(
                #     image_faded.copy(),
                #     data_sample.gt_instances,
                #     data_sample,
                #     classes,
                #     palette,
                #     draw_majmin_lines=False,
                #     draw_PCA_basis=False,
                #     draw_points=True,
                # )

                canvas_img_data = self._draw_instances(
                    raw_img_data.copy(),
                    data_sample.pred_instances_canvas,
                    data_sample,
                    classes,
                    palette,
                    draw_majmin_lines=False,
                    draw_PCA_basis=False,
                )

                pred_img_data = self._draw_instances(
                    raw_img_data.copy(),
                    data_sample.pred_instances,
                    data_sample,
                    classes,
                    palette,
                    draw_majmin_lines=False,
                    draw_PCA_basis=False,
                )

                # pred_intermediate_0_img_data = self._draw_instances(
                #     image_faded.copy(),
                #     data_sample.gt_instances,
                #     data_sample,
                #     classes,
                #     palette,
                #     draw_majmin_lines=False,
                #     draw_PCA_basis=False,
                #     draw_points=False,
                #     draw_intermediate=0,
                # )
                # pred_intermediate_1_img_data = self._draw_instances(
                #     image_faded.copy(),
                #     data_sample.gt_instances,
                #     data_sample,
                #     classes,
                #     palette,
                #     draw_majmin_lines=False,
                #     draw_PCA_basis=False,
                #     draw_points=False,
                #     draw_intermediate=1,
                # )
                # pred_intermediate_2_img_data = self._draw_instances(
                #     image_faded.copy(),
                #     data_sample.gt_instances,
                #     data_sample,
                #     classes,
                #     palette,
                #     draw_majmin_lines=False,
                #     draw_PCA_basis=False,
                #     draw_points=False,
                #     draw_intermediate=2,
                # )
                # pred_intermediate_3_img_data = self._draw_instances(
                #     image_faded.copy(),
                #     data_sample.gt_instances,
                #     data_sample,
                #     classes,
                #     palette,
                #     draw_majmin_lines=False,
                #     draw_PCA_basis=False,
                #     draw_points=False,
                #     draw_intermediate=3,
                # )
                # pred_intermediate_4_img_data = self._draw_instances(
                #     image_faded.copy(),
                #     data_sample.gt_instances,
                #     data_sample,
                #     classes,
                #     palette,
                #     draw_majmin_lines=False,
                #     draw_PCA_basis=False,
                #     draw_points=False,
                #     draw_intermediate=4,
                # )
                # pred_intermediate_5_img_data = self._draw_instances(
                #     image_faded.copy(),
                #     data_sample.gt_instances,
                #     data_sample,
                #     classes,
                #     palette,
                #     draw_majmin_lines=False,
                #     draw_PCA_basis=False,
                #     draw_points=False,
                #     draw_intermediate=5,
                # )
                # pred_intermediate_6_img_data = self._draw_instances(
                #     image_faded.copy(),
                #     data_sample.gt_instances,
                #     data_sample,
                #     classes,
                #     palette,
                #     draw_majmin_lines=False,
                #     draw_PCA_basis=False,
                #     draw_points=False,
                #     draw_intermediate=6,
                # )

            # plot sampling area
            # if 'anatomical_pole_area' in data_sample.keys():
            # if True:
            #     gt_img_data = self.plot_poles(gt_img_data, data_sample.anatomical_pole_area)

        # plot extreme point candidat using markers "draw_points" of different color per area
        # create func
        # if 'gt_instances' in data_sample:

        #     if data_sample.gt_instances.bboxes.sum() > 0:

        #         scale_y, scale_x = data_sample.scale_factor  # Unpack scale factors
        #         # Apply scaling to the extreme points
        #         data_sample.gt_instances.anatomical_pole_pools = \
        #             data_sample.gt_instances.anatomical_pole_pools / torch.tensor([scale_x, scale_y], \
        #                 device=data_sample.gt_instances.anatomical_pole_pools.device)

        gt_img_data_axis_pool = self.plot_poles(
            gt_img_data_axis, data_sample.anatomical_pole_area
        )

        # list n_inst, list 4, shape(h, w)
        # data_sample.anatomical_pole_area[0][0].shape
        # breakpoint()

        # drawn_img = np.concatenate((raw_img_data, gt_img_data), axis=1)
        # drawn_img = np.concatenate((raw_img_data, gt_img_data, canvas_img_data, pred_img_data), axis=1)
        # top = np.concatenate((raw_img_data, gt_img_data), axis=1)
        # bottom = np.concatenate((canvas_img_data, pred_img_data), axis=1)
        # drawn_img = np.concatenate((top, bottom), axis=0)

        drawn_img = np.concatenate(
            (gt_img_data, canvas_img_data, pred_img_data), axis=1
        )

        self.set_image(drawn_img)

        import matplotlib.colors as mcolors

        def draw_border(img, color="#ffa756", px=4, crop_bottom=0, inplace=False):
            out = img if inplace else img.copy()
            h, w = out.shape[:2]
            if crop_bottom > 0:
                out = out[: h - crop_bottom, :]
            if isinstance(color, str):  # hex string
                color = hex_to_rgb255(color)
            c = np.array(color, dtype=out.dtype)

            if out.ndim == 2:  # grayscale
                c = c if np.isscalar(c) else c[0]
                out[:px, :] = c
                out[-px:, :] = c
                out[:, :px] = c
                out[:, -px:] = c
            else:
                # out[:px, :] = c
                out[-px:, :] = c
                # out[:, :px] = c
                # out[:, -px:] = c
            return out

        def hex_to_rgb255(hx: str):
            """Return tuple of ints 0–255."""
            return tuple(int(round(v * 255)) for v in mcolors.to_rgb(hx))

        # colors = ["#ffa756", "#d58a94", "#5170d7"]
        gt_with_border = draw_border(
            gt_img_data, color="#5170d7", px=15, crop_bottom=30
        )

        # pred_intermediate_0_img_data = draw_border(
        #     pred_intermediate_0_img_data, color="#5170d7", px=15, crop_bottom=30
        # )
        # pred_intermediate_1_img_data = draw_border(
        #     pred_intermediate_1_img_data, color="#5170d7", px=15, crop_bottom=30
        # )
        # pred_intermediate_2_img_data = draw_border(
        #     pred_intermediate_2_img_data, color="#5170d7", px=15, crop_bottom=30
        # )
        # pred_intermediate_3_img_data = draw_border(
        #     pred_intermediate_3_img_data, color="#5170d7", px=15, crop_bottom=30
        # )
        # pred_intermediate_4_img_data = draw_border(
        #     pred_intermediate_4_img_data, color="#5170d7", px=15, crop_bottom=30
        # )
        # pred_intermediate_5_img_data = draw_border(
        #     pred_intermediate_5_img_data, color="#5170d7", px=15, crop_bottom=30
        # )
        # pred_intermediate_6_img_data = draw_border(
        #     pred_intermediate_6_img_data, color="#5170d7", px=15, crop_bottom=30
        # )
        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
            # mmcv.imwrite(
            #     gt_with_border[..., ::-1],
            #     ".".join(out_file.split(".")[:-1]) + "_raw_mask_" + ".png",
            # )
            # mmcv.imwrite(
            #     gt_img_data_axis[..., ::-1],
            #     ".".join(out_file.split(".")[:-1]) + "_mask_axis_" + ".png",
            # )
            # mmcv.imwrite(
            #     gt_img_data_axis_pool[..., ::-1],
            #     ".".join(out_file.split(".")[:-1]) + "_mask_axis_pool_" + ".png",
            # )
            # mmcv.imwrite(
            #     gt_img_data_axis_point[..., ::-1],
            #     ".".join(out_file.split(".")[:-1]) + "_mask_point_" + ".png",
            # )
            # mmcv.imwrite(
            #     pred_intermediate_0_img_data[..., ::-1],
            #     ".".join(out_file.split(".")[:-1]) + "_intermediate_0_" + ".png",
            # )
            # mmcv.imwrite(
            #     pred_intermediate_1_img_data[..., ::-1],
            #     ".".join(out_file.split(".")[:-1]) + "_intermediate_1_" + ".png",
            # )
            # mmcv.imwrite(
            #     pred_intermediate_2_img_data[..., ::-1],
            #     ".".join(out_file.split(".")[:-1]) + "_intermediate_2_" + ".png",
            # )
            # mmcv.imwrite(
            #     pred_intermediate_3_img_data[..., ::-1],
            #     ".".join(out_file.split(".")[:-1]) + "_intermediate_3_" + ".png",
            # )
            # mmcv.imwrite(
            #     pred_intermediate_4_img_data[..., ::-1],
            #     ".".join(out_file.split(".")[:-1]) + "_intermediate_4_" + ".png",
            # )
            # mmcv.imwrite(
            #     pred_intermediate_5_img_data[..., ::-1],
            #     ".".join(out_file.split(".")[:-1]) + "_intermediate_5_" + ".png",
            # )
            # mmcv.imwrite(
            #     pred_intermediate_6_img_data[..., ::-1],
            #     ".".join(out_file.split(".")[:-1]) + "_intermediate_6_" + ".png",
            # )

            # quit()
            # breakpoint()
            # NOTE here write each output img
            # mmcv.imwrite(gt_with_border[..., ::-1], out_file)
        # else:
        #     self.add_image(name, drawn_img, step)

        # from PIL import Image

        # rgba_instance_imgs = self.extract_instances_with_transparency(
        #     gt_img_data,
        #     data_sample.gt_instances,
        #     data_sample.anatomical_pole_area,
        #     with_pool=True,
        # )
        # img_dir = next(iter(self._vis_backends.values()))._img_save_dir

        # for i, rgba_instance_img in enumerate(rgba_instance_imgs):
        #     # self.add_image(f"{name}_instance_{i}", rgba_instance_img, step)
        #     # Image.fromarray(image).save(self._vis_backends[0]._img_save_dir, save_file_name))
        #     # imageio.imwrite(f"{img_dir}/{name}_instance_{i}", image)
        #     Image.fromarray(rgba_instance_img).save(
        #         f"{img_dir}/{name}_{step}_instance_{i}.png", dpi=(300, 300)
        #     )
