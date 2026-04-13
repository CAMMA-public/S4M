from mmdet.evaluation.metrics.semseg_metric import SemSegMetric
from mmdet.registry import METRICS
import os.path as osp
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from mmcv import imwrite
from PIL import Image


@METRICS.register_module()
class InstanceSegMetric(SemSegMetric):
    # def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
    #     """Process one batch of data and data_samples.

    #     The processed results should be stored in ``self.results``, which will
    #     be used to compute the metrics when all batches have been processed.

    #     Args:
    #         data_batch (dict): A batch of data from the dataloader.
    #         data_samples (Sequence[dict]): A batch of outputs from the model.
    #     """
    #     num_classes = len(self.dataset_meta['classes'])
    #     for data_sample in data_samples:
    #         # shape (h, w)
    #         pred_label = data_sample['pred_sem_seg']['sem_seg'].squeeze()
    #         # format_only always for test dataset without ground truth
    #         if not self.format_only:
    #             # shape (h, w)
    #             label = data_sample['gt_sem_seg']['sem_seg'].squeeze().to(
    #                 pred_label)
    #             ignore_index = data_sample['pred_sem_seg'].get(
    #                 'ignore_index', 255)
    #             self.results.append(
    #                 self._compute_pred_stats(pred_label, label, num_classes,
    #                                          ignore_index))

    #         # format_result
    #         if self.output_dir is not None:
    #             basename = osp.splitext(osp.basename(
    #                 data_sample['img_path']))[0]
    #             png_filename = osp.abspath(
    #                 osp.join(self.output_dir, f'{basename}.png'))
    #             output_mask = pred_label.cpu().numpy()
    #             output = Image.fromarray(output_mask.astype(np.uint8))
    #             imwrite(output, png_filename, backend_args=self.backend_args)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['sem_seg'].squeeze()
            if not self.format_only:
                label = data_sample['gt_sem_seg']['sem_seg'].squeeze().to(pred_label)
                ignore_index = data_sample['pred_sem_seg'].get('ignore_index', 255)
                self.results.append(
                    self._compute_instance_stats(pred_label, label, ignore_index))

            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(data_sample['img_path']))[0]
                png_filename = osp.abspath(osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                output = Image.fromarray(output_mask.astype(np.uint8))
                imwrite(output, png_filename, backend_args=self.backend_args)

    def _compute_instance_stats(self, pred_mask, gt_mask, ignore_index=255):
        instance_ids = torch.unique(gt_mask)
        instance_ids = instance_ids[instance_ids != ignore_index]

        total_tp = 0
        total_fp = 0
        total_fn = 0
        ious = []
        accs = []

        for inst_id in instance_ids:
            gt_instance = (gt_mask == inst_id)
            pred_instance = (pred_mask == inst_id)

            inter = (gt_instance & pred_instance).sum().item()
            union = (gt_instance | pred_instance).sum().item()
            tp = inter
            fp = pred_instance.sum().item() - tp
            fn = gt_instance.sum().item() - tp

            iou = inter / union if union > 0 else 0
            acc = tp / gt_instance.sum().item() if gt_instance.sum().item() > 0 else 0

            total_tp += tp
            total_fp += fp
            total_fn += fn
            ious.append(iou)
            accs.append(acc)

        total_pixels = (gt_mask != ignore_index).sum().item()
        correct_pixels = (gt_mask == pred_mask).masked_fill(gt_mask == ignore_index, 0).sum().item()
        aAcc = correct_pixels / total_pixels if total_pixels > 0 else 0

        return dict(
            ious=ious,
            accs=accs,
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            aAcc=aAcc
        )



    def compute_metrics(self, results: list) -> Dict[str, float]:
        if self.format_only:
            logger = MMLogger.get_current_instance()
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        all_ious, all_accs, tp, fp, fn, aAcc_total = [], [], 0, 0, 0, 0
        for r in results:
            all_ious.extend(r['ious'])
            all_accs.extend(r['accs'])
            tp += r['tp']
            fp += r['fp']
            fn += r['fn']
            aAcc_total += r['aAcc']

        eps = 1e-6
        mIoU = np.mean(all_ious) if all_ious else 0
        mAcc = np.mean(all_accs) if all_accs else 0
        mDice = 2 * tp / (2 * tp + fp + fn + eps)
        mPrecision = tp / (tp + fp + eps)
        mRecall = tp / (tp + fn + eps)
        mFscore = 2 * mPrecision * mRecall / (mPrecision + mRecall + eps)
        aAcc = aAcc_total / len(results)

        return dict(
            aAcc=round(aAcc * 100, 2),
            mIoU=round(mIoU * 100, 2),
            mDice=round(mDice * 100, 2),
            mAcc=round(mAcc * 100, 2),
            mPrecision=round(mPrecision * 100, 2),
            mRecall=round(mRecall * 100, 2),
            mFscore=round(mFscore * 100, 2)
        )
