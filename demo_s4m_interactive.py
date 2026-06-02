#!/usr/bin/env python
"""Interactive S4M demo driven by four major/minor clicks."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-s4m-demo")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.nn.functional as torch_f
from mmcv.transforms import Compose
from matplotlib.widgets import Button, CheckButtons
from mmdet.apis import init_detector
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import register_all_modules
from mmengine.config import Config

from S4M.models.utils.custom_functional import multi_head_attention_forward
from S4M.models.task_modules.prior_generators.prompt_encoder import EmbeddingIndex

DEFAULT_IMAGE = "sample_dataset/sample_images/test.MMOTU_2d__00003.png"
DEFAULT_CONFIG = "S4M/configs/S4M/mmotu_majmin.py"
DEFAULT_CHECKPOINT = "UltraS4M_majmin.pth"

MAJOR_COLOR = "#1f77b4"
MINOR_COLOR = "#aec7e8"
POS_COLOR = "lime"
NEG_COLOR = "red"
MASK_COLOR = np.array([81, 112, 215], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Image to segment.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="S4M config path.")
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="S4M checkpoint path."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference, e.g. cpu or cuda:0.",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def load_model(config, checkpoint, device):
    torch_f.multi_head_attention_forward = multi_head_attention_forward
    cfg_options = {"model.num_mask_refinements": 0}
    model = init_detector(
        str(config),
        str(checkpoint),
        device=device,
        cfg_options=cfg_options,
    )
    model.eval()
    return model


def build_pipelines(config):
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config))
    image_steps = []
    pack_step = None
    annotation_section = False
    for step in cfg.test_pipeline:
        if step["type"] == "LoadAnnotations":
            annotation_section = True
            continue
        if step["type"].startswith("Pack"):
            pack_step = step
            continue
        if not annotation_section:
            image_steps.append(step)

    if pack_step is None:
        raise RuntimeError("Could not find a Pack* transform in the test pipeline.")
    return Compose(image_steps), Compose([pack_step])


def load_pipeline_image(image_pipeline, image_path):
    data = image_pipeline(dict(img_path=str(image_path), img_id=0))
    if data is None:
        raise RuntimeError(f"Could not load image through test pipeline: {image_path}")
    return data


def pair_length(pair):
    a, b = np.asarray(pair[0]), np.asarray(pair[1])
    return float(np.linalg.norm(a - b))


def order_major_minor(points):
    first_pair = points[:2]
    second_pair = points[2:4]
    if pair_length(first_pair) >= pair_length(second_pair):
        return np.asarray(first_pair + second_pair, dtype=np.float32)
    return np.asarray(second_pair + first_pair, dtype=np.float32)


def transform_points(image_data, points):
    points = np.asarray(points, dtype=np.float32)
    homography = image_data.get("homography_matrix")
    if homography is not None:
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.concatenate([points, ones], axis=1)
        transformed = points_h @ np.asarray(homography, dtype=np.float32).T
        return transformed[:, :2] / transformed[:, 2:3]

    scale = np.asarray(image_data["scale_factor"], dtype=np.float32)
    return points * scale


def bbox_from_points(points, width, height, padding=8.0):
    pts = np.asarray(points, dtype=np.float32)
    x1 = max(0.0, float(pts[:, 0].min() - padding))
    y1 = max(0.0, float(pts[:, 1].min() - padding))
    x2 = min(float(width - 1), float(pts[:, 0].max() + padding))
    y2 = min(float(height - 1), float(pts[:, 1].max() + padding))
    if x2 <= x1:
        x2 = min(float(width - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height - 1), y1 + 1.0)
    return np.array([[x1, y1, x2, y2]], dtype=np.float32)


def synthetic_mask(points, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) >= 3:
        hull = cv2.convexHull(np.round(pts).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
    else:
        x1, y1, x2, y2 = bbox_from_points(pts, width, height)[0].astype(np.int32)
        mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    if mask.sum() == 0:
        x, y = np.round(pts[0]).astype(np.int32)
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        mask[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)] = 1
    return BitmapMasks(mask[None], height, width)


def build_data(model, pack_pipeline, image_data, majmin_points, correction_points):
    ori_h, ori_w = image_data["ori_shape"]
    img_h, img_w = image_data["img_shape"]

    scaled_majmin = transform_points(image_data, order_major_minor(majmin_points))
    scaled_corrections = []
    correction_types = []
    for x, y, point_type in correction_points:
        scaled_corrections.append(transform_points(image_data, [(x, y)])[0])
        correction_types.append(point_type)

    results = image_data.copy()
    results["gt_bboxes"] = bbox_from_points(scaled_majmin, img_w, img_h)
    results["gt_bboxes_labels"] = np.zeros(1, dtype=np.int64)
    results["gt_masks"] = synthetic_mask(majmin_points, ori_w, ori_h)
    results["anatomical_pole_pools"] = scaled_majmin[None].astype(np.float32)

    packed = pack_pipeline(results)
    gt_instances = packed["data_samples"].gt_instances
    if scaled_corrections:
        gt_instances.interactive_points = torch.from_numpy(
            np.asarray(scaled_corrections, dtype=np.float32)[None]
        )
        gt_instances.interactive_points_types = torch.from_numpy(
            np.asarray(correction_types, dtype=np.int64)[None]
        )

    data = dict(inputs=[packed["inputs"]], data_samples=[packed["data_samples"]])
    return model.data_preprocessor(data, False)


@torch.no_grad()
def predict(model, pack_pipeline, image_data, majmin_points, correction_points):
    processed = build_data(
        model, pack_pipeline, image_data, majmin_points, correction_points
    )
    batch_inputs = processed["inputs"]
    batch_data_samples = processed["data_samples"]
    return model.predict(batch_inputs, batch_data_samples, rescale=True)[0]


def mask_to_numpy(data_sample):
    masks = data_sample.pred_instances.masks
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if masks.size == 0:
        return None
    return masks[0].astype(bool)


def draw_overlay(image_rgb, mask):
    canvas = image_rgb.astype(np.float32).copy()
    if mask is None:
        return canvas.astype(np.uint8)
    overlay = canvas.copy()
    overlay[mask] = overlay[mask] * 0.45 + MASK_COLOR * 0.55
    return overlay.astype(np.uint8)


class InteractiveDemo:
    def __init__(self, model, image_bgr, image_data, pack_pipeline):
        self.model = model
        self.image_bgr = image_bgr
        self.image_rgb = mmcv.bgr2rgb(image_bgr)
        self.image_data = image_data
        self.pack_pipeline = pack_pipeline
        self.major_minor_points = []
        self.correction_points = []
        self.current_mode = "Positive"
        self.mask = None
        self.busy = False
        self.updating_mode_buttons = False

        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.fig.subplots_adjust(bottom=0.22)
        self.image_artist = self.ax.imshow(self.image_rgb)
        self.ax.set_axis_off()
        self.status = self.fig.text(0.02, 0.96, "", ha="left", va="top")

        self.fig.text(0.02, 0.175, "Select type", ha="left", va="bottom")
        self.mode_ax = self.fig.add_axes([0.02, 0.04, 0.16, 0.12])
        self.mode_buttons = CheckButtons(
            self.mode_ax, ("Positive", "Negative"), (True, False)
        )
        self.mode_buttons.on_clicked(self.on_mode)

        self.segment_button = self.add_button(
            [0.22, 0.055, 0.16, 0.055], "Segment / Update"
        )
        self.undo_button = self.add_button([0.41, 0.055, 0.13, 0.055], "Undo")
        self.reset_button = self.add_button([0.57, 0.055, 0.13, 0.055], "Reset")
        self.quit_button = self.add_button([0.73, 0.055, 0.13, 0.055], "Quit")

        self.segment_button.on_clicked(self.on_segment)
        self.undo_button.on_clicked(self.on_undo)
        self.reset_button.on_clicked(self.on_reset)
        self.quit_button.on_clicked(self.on_quit)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.redraw()

    def add_button(self, rect, label):
        return Button(self.fig.add_axes(rect), label)

    def on_mode(self, label):
        if self.updating_mode_buttons:
            return
        self.current_mode = label
        target_states = (label == "Positive", label == "Negative")
        current_states = self.mode_buttons.get_status()
        self.updating_mode_buttons = True
        try:
            for idx, (current, target) in enumerate(zip(current_states, target_states)):
                if current != target:
                    self.mode_buttons.set_active(idx)
        finally:
            self.updating_mode_buttons = False
        self.redraw()

    def on_click(self, event):
        if (
            self.busy
            or event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return
        x = float(event.xdata)
        y = float(event.ydata)
        height, width = self.image_rgb.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return

        if len(self.major_minor_points) < 4:
            self.major_minor_points.append((x, y))
            if len(self.major_minor_points) == 4:
                self.redraw()
                self.fig.canvas.flush_events()
                self.on_segment()
                return
        else:
            point_type = (
                EmbeddingIndex.POS.value
                if self.current_mode == "Positive"
                else EmbeddingIndex.NEG.value
            )
            self.correction_points.append((x, y, point_type))
        self.redraw()

    def on_segment(self, _event=None):
        if self.busy or len(self.major_minor_points) < 4:
            self.redraw("Click four major/minor points first.")
            return
        self.busy = True
        self.redraw("Running S4M...")
        self.fig.canvas.flush_events()
        try:
            result = predict(
                self.model,
                self.pack_pipeline,
                self.image_data,
                self.major_minor_points,
                self.correction_points,
            )
            self.mask = mask_to_numpy(result)
            self.redraw("Segmentation updated.")
        except Exception as exc:
            self.redraw(f"S4M failed: {exc}")
            raise
        finally:
            self.busy = False

    def on_undo(self, _event=None):
        if self.busy:
            return
        if self.correction_points:
            self.correction_points.pop()
        elif self.major_minor_points:
            self.major_minor_points.pop()
            self.mask = None
        self.redraw()

    def on_reset(self, _event=None):
        if self.busy:
            return
        self.major_minor_points = []
        self.correction_points = []
        self.mask = None
        self.redraw()

    def on_quit(self, _event=None):
        plt.close(self.fig)

    def redraw(self, message=None):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.imshow(draw_overlay(self.image_rgb, self.mask))
        if self.mask is not None:
            contours, _ = cv2.findContours(
                self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                contour = contour.squeeze(1)
                self.ax.plot(contour[:, 0], contour[:, 1], color="white", linewidth=1.5)

        self.draw_initial_points()
        self.draw_correction_points()
        self.status.set_text(message or self.default_status())
        self.fig.canvas.draw_idle()

    def draw_initial_points(self):
        pts = self.major_minor_points
        for idx, (x, y) in enumerate(pts):
            color = MAJOR_COLOR if len(pts) == 4 and idx < 2 else MINOR_COLOR
            if len(pts) == 4:
                ordered = order_major_minor(pts)
                color = (
                    MAJOR_COLOR
                    if any(np.allclose([x, y], p) for p in ordered[:2])
                    else MINOR_COLOR
                )
            self.ax.scatter([x], [y], s=140, c="white", edgecolors="none", zorder=4)
            self.ax.scatter([x], [y], s=85, c=color, edgecolors="none", zorder=5)
            label = "M" if color == MAJOR_COLOR else "m"
            self.ax.text(
                x, y + 10, label, color="navy", ha="center", va="center", zorder=6
            )

        if len(pts) >= 2:
            self.ax.plot(
                [pts[0][0], pts[1][0]],
                [pts[0][1], pts[1][1]],
                color="white",
                linewidth=1.4,
                linestyle="-",
            )
        if len(pts) >= 4:
            ordered = order_major_minor(pts)
            self.ax.plot(
                [ordered[0, 0], ordered[1, 0]],
                [ordered[0, 1], ordered[1, 1]],
                color=MAJOR_COLOR,
                linewidth=2.0,
                linestyle="-",
            )
            self.ax.plot(
                [ordered[2, 0], ordered[3, 0]],
                [ordered[2, 1], ordered[3, 1]],
                color="white",
                linewidth=2.0,
                linestyle="--",
            )
            self.ax.plot(
                [ordered[2, 0], ordered[3, 0]],
                [ordered[2, 1], ordered[3, 1]],
                color=MINOR_COLOR,
                linewidth=1.4,
                linestyle="--",
            )
        elif len(pts) == 3:
            self.ax.scatter([pts[2][0]], [pts[2][1]], s=85, c=MINOR_COLOR, zorder=5)

    def draw_correction_points(self):
        for x, y, point_type in self.correction_points:
            color = POS_COLOR if point_type == EmbeddingIndex.POS.value else NEG_COLOR
            self.ax.scatter([x], [y], s=130, c="white", edgecolors="none", zorder=7)
            self.ax.scatter([x], [y], s=80, c=color, edgecolors="none", zorder=8)

    def default_status(self):
        if len(self.major_minor_points) < 4:
            return (
                f"Click major/minor point {len(self.major_minor_points) + 1}/4. "
                "Points 1-2 and 3-4 form pairs."
            )
        return (
            f"Mode: {self.current_mode}. Click correction points, then Segment. "
            f"Corrections: {len(self.correction_points)}"
        )


def main():
    args = parse_args()
    image_path = resolve_path(args.image)
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)

    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    image_bgr = mmcv.imread(str(image_path), channel_order="bgr")
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    image_pipeline, pack_pipeline = build_pipelines(config_path)
    image_data = load_pipeline_image(image_pipeline, image_path)
    model = load_model(config_path, checkpoint_path, args.device)
    demo = InteractiveDemo(model, image_bgr, image_data, pack_pipeline)
    plt.show()
    return demo


if __name__ == "__main__":
    main()
