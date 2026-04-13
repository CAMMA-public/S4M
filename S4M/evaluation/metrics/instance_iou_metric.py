import copy
import warnings
from collections import OrderedDict
from typing import List, Optional, Sequence, Union, Any, Tuple, Dict

import numpy as np
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from pycocotools import mask as maskUtils
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
import torch
from skimage import measure, morphology
import json


@METRICS.register_module()
class InstanceIoUMetric(CocoMetric):
    """
    Calcule uniquement le mIoU instance-wise (global + par classe) en supposant
    que `results` est une séquence de tuples (gt, pred) déjà appariés 1-1,
    et que gt['mask'] / pred['mask'] sont des RLE dict pycocotools.
    - Classe par instance prise depuis gt['bbox_label'].
    """

    default_prefix = "custom"

    def __init__(self, *args, per_class: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_class = per_class

    @staticmethod
    def _norm_rle(r: Any) -> dict:
        """Assure un RLE compatible pycocotools (counts en str)."""
        if isinstance(r.get("counts", None), bytes):
            r = dict(r)
            r["counts"] = r["counts"].decode()
        return r

    @staticmethod
    def _safe_iou(pred_rle: dict, gt_rle: dict) -> float:
        iou = maskUtils.iou([pred_rle], [gt_rle], [0])[0, 0]
        return float(iou) if np.isfinite(iou) else 0.0

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample["pred_instances"]
            result["img_id"] = data_sample["img_id"]
            result["bboxes"] = pred["bboxes"].cpu().numpy()
            result["scores"] = pred["scores"].cpu().numpy()
            result["labels"] = pred["labels"].cpu().numpy()
            # encode mask to RLE
            if "masks" in pred:
                result["masks"] = (
                    encode_mask_results(pred["masks"].detach().cpu().numpy())
                    if isinstance(pred["masks"], torch.Tensor)
                    else pred["masks"]
                )
            # some detectors use different scores for bbox and mask
            if "mask_scores" in pred:
                result["mask_scores"] = pred["mask_scores"].cpu().numpy()

            # parse gt
            gt = dict()
            gt["width"] = data_sample["ori_shape"][1]
            gt["height"] = data_sample["ori_shape"][0]
            gt["img_id"] = data_sample["img_id"]
            gt["labels"] = data_sample["gt_instances"]["labels"].detach().cpu().numpy()
            # BitmapMasks -> rle
            gt["masks"] = encode_mask_results(
                data_sample["gt_instances"]["masks"].to_tensor(
                    pred["masks"].dtype, device="cpu"
                )
            )

            # add converted result to the results list
            self.results.append((gt, result))

    def compute_metrics(self, results: Sequence[Tuple[dict, dict]]) -> Dict[str, float]:
        classes_all = self.dataset_meta.get("classes", None)
        if classes_all is None:
            raise ValueError('dataset_meta["classes"] est requis.')
        # filtre éventuel "background"
        classes = [c for c in classes_all if str(c).lower() != "background"]

        # map des ids d’origine -> indices compacts
        idx_map = {}
        j = 0
        for i, name in enumerate(classes_all):
            if str(name).lower() != "background":
                idx_map[i] = j
                j += 1
        C = len(classes)

        iou_sum, n_pairs = 0.0, 0
        per_sum = [0.0] * C
        per_cnt = [0] * C

        for gt, pred in results:
            if "masks" not in gt or "masks" not in pred:
                continue
            gt_masks = gt["masks"]  # liste/array de RLE dicts
            pred_masks = pred["masks"]  # liste/array de RLE dicts
            gt_labels = gt.get("labels", None)

            N = min(len(gt_masks), len(pred_masks))
            if N == 0:
                continue

            for k in range(N):
                gt_rle = self._norm_rle(gt_masks[k])
                pr_rle = self._norm_rle(pred_masks[k])
                iou = self._safe_iou(pr_rle, gt_rle)

                iou_sum += iou
                n_pairs += 1

                if self.per_class and gt_labels is not None:
                    c = int(gt_labels[k])
                    # c = idx_map.get(c_raw, None)
                    if c is not None and 0 <= c:
                        per_sum[c] += iou
                        per_cnt[c] += 1

        out = OrderedDict()
        out["mIoU_instance"] = (iou_sum / n_pairs) if n_pairs else 0.0

        if self.per_class:
            mean_per_class = 0
            for i, name in enumerate(classes):
                out[f"mIoU_instance/{name}"] = (
                    (per_sum[i] / per_cnt[i]) if per_cnt[i] else 0.0
                )
                mean_per_class += out[f"mIoU_instance/{name}"]
            out["mIoU_instance/mean"] = mean_per_class / len(classes)
        return out


@METRICS.register_module()
class InstanceIoUConcavityMetric(InstanceIoUMetric):
    def _concavity(self, rle: dict) -> float:
        """
        Compute concavity = 1 - area(mask)/area(convex_hull(mask))
        rle: dict COCO RLE
        """
        mask = maskUtils.decode(rle).astype(bool)  # (H, W)
        area = mask.sum()
        if area == 0:
            return 0.0

        hull = morphology.convex_hull_image(mask)
        hull_area = hull.sum()
        if hull_area == 0:
            return 0.0

        return 1.0 - (area / hull_area)

    def compute_metrics(
        self,
        results: Sequence[Tuple[dict, dict]],
        dump_json_path=None,
    ) -> Dict[str, float]:
        classes_all = self.dataset_meta.get("classes", None)
        if classes_all is None:
            raise ValueError('dataset_meta["classes"] est requis.')
        # filtre éventuel "background"
        classes = [c for c in classes_all if str(c).lower() != "background"]

        # map des ids d’origine -> indices compacts
        idx_map = {}
        j = 0
        for i, name in enumerate(classes_all):
            if str(name).lower() != "background":
                idx_map[i] = j
                j += 1
        C = len(classes)

        iou_sum, n_pairs = 0.0, 0
        per_sum = [0.0] * C
        per_cnt = [0] * C

        per_instance = []  # list[dict]

        for gt, pred in results:
            if "masks" not in gt or "masks" not in pred:
                continue
            gt_masks = gt["masks"]  # liste/array de RLE dicts
            pred_masks = pred["masks"]  # liste/array de RLE dicts
            gt_labels = gt.get("labels", None)

            N = min(len(gt_masks), len(pred_masks))
            if N == 0:
                continue

            for k in range(N):
                gt_rle = self._norm_rle(gt_masks[k])
                pr_rle = self._norm_rle(pred_masks[k])
                iou = self._safe_iou(pr_rle, gt_rle)

                conc = self._concavity(gt_rle)  # <- assume existante

                iou_sum += iou
                n_pairs += 1

                if self.per_class and gt_labels is not None:
                    c = int(gt_labels[k])
                    # c = idx_map.get(c_raw, None)
                    if c is not None and 0 <= c:
                        per_sum[c] += iou
                        per_cnt[c] += 1

                per_instance.append(
                    {
                        "mIoU": float(iou),
                        "gt_concavity": float(conc),
                    }
                )

        out = OrderedDict()
        out["mIoU_instance"] = (iou_sum / n_pairs) if n_pairs else 0.0

        # if self.per_class:
        #     mean_per_class = 0
        #     for i, name in enumerate(classes):
        #         out[f"mIoU_instance/{name}"] = (
        #             (per_sum[i] / per_cnt[i]) if per_cnt[i] else 0.0
        #         )
        #         mean_per_class += out[f"mIoU_instance/{name}"]
        #     out["mIoU_instance/mean"] = mean_per_class / len(classes)

        if self.per_class:
            mean_per_class = 0
            valid_classes = 0
            for i, name in enumerate(classes):
                if per_cnt[i]:
                    val = per_sum[i] / per_cnt[i]
                    out[f"mIoU_instance/{name}"] = val
                    mean_per_class += val
                    valid_classes += 1
                else:
                    out[f"mIoU_instance/{name}"] = 0.0
            out["mIoU_instance/mean"] = (
                mean_per_class / valid_classes if valid_classes > 0 else 0.0
            )

        if dump_json_path is not None:
            with open(dump_json_path, "w", encoding="utf-8") as f:
                json.dump(per_instance, f, ensure_ascii=False, indent=2)

        return out
