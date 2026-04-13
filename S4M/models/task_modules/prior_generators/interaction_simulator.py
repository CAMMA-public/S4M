import torch
from S4M.models.task_modules.prior_generators.prompt_encoder import EmbeddingIndex


class InteractionSimulator:
    _instance = None

    def __new__(cls, threshold=0.5, mult_factor=4):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.threshold = threshold
            cls._instance.mult_factor = mult_factor
            cls._instance.reset()
        return cls._instance

    def reset(self):
        self.error_maps = None
        self.yx = None
        self.point_types = None

    def compute_error_maps(self, mask_preds, gt_masks):
        pred_bin = (mask_preds.sigmoid() >= self.threshold).squeeze(1)
        gt_bin = gt_masks.bool()

        TP = gt_bin & pred_bin
        TN = ~gt_bin & ~pred_bin
        FP = ~gt_bin & pred_bin
        FN = gt_bin & ~pred_bin

        self.error_maps = {
            "TP": TP.float(),
            "TN": TN.float(),
            "FP": FP.float(),
            "FN": FN.float(),
        }
        return self.error_maps

    def sample_prompt_point(self):
        if self.error_maps is None:
            raise RuntimeError("Call compute_error_maps first.")

        fp = self.error_maps["FP"]
        fn = self.error_maps["FN"]
        tp = self.error_maps["TP"]

        batch_size = fp.shape[0]
        prompts = []
        types = []

        for i in range(batch_size):
            # Try FP ∪ FN
            error_mask = (fp[i] + fn[i]) > 0
            coords = torch.nonzero(error_mask)

            if len(coords) > 0:
                idx = torch.randint(len(coords), (1,))
                yx = tuple(coords[idx][0].tolist())
                point_type = (
                    EmbeddingIndex.POS.value
                    if fn[i][yx] > 0
                    else EmbeddingIndex.NEG.value
                )  # positive or negative
            else:
                coords = torch.nonzero(tp[i])
                if len(coords) == 0:
                    prompts.append(torch.tensor([-1, -1]))
                    types.append(torch.tensor([-1]))
                    continue
                idx = torch.randint(len(coords), (1,))
                yx = tuple(coords[idx][0].tolist())
                point_type = EmbeddingIndex.POS.value  # foreground

            scaled_yx = torch.tensor(yx) * self.mult_factor
            prompts.append(scaled_yx)
            types.append(torch.tensor([point_type]))

        # [N, 1, 2] and [N, 1, 1]
        new_prompts = torch.stack(prompts).unsqueeze(1)[..., [1, 0]]
        new_types = torch.stack(types).unsqueeze(1)

        if self.yx is None:
            self.yx = new_prompts  # [N, 1, 2]
            self.point_types = new_types  # [N, 1, 1]
        else:
            self.yx = torch.cat([self.yx, new_prompts], dim=1)  # [N, K, 2]
            self.point_types = torch.cat(
                [self.point_types, new_types], dim=1
            )  # [N, K, 1]

        return new_prompts, new_types

    def get_next_prompt(self, mask_preds, gt_masks):
        self.compute_error_maps(mask_preds, gt_masks)
        return self.sample_prompt_point()

    def get_next_prompt_from_binary(self, mask_preds, gt_masks):
        pred_bin = mask_preds.bool()
        gt_bin = gt_masks.bool()

        TP = gt_bin & pred_bin
        TN = ~gt_bin & ~pred_bin
        FP = ~gt_bin & pred_bin
        FN = gt_bin & ~pred_bin

        self.error_maps = {
            "TP": TP.float(),
            "TN": TN.float(),
            "FP": FP.float(),
            "FN": FN.float(),
        }

        return self.sample_prompt_point()
