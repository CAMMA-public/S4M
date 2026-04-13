_base_ = [
    "./S4M_majmin.py",
]

data_root = "./DATA/EndoSAM/"

metainfo = {
    "classes": (
        "background",
        "cystic_plate",
        "calot_triangle",
        "cystic_artery",
        "cystic_duct",
        "gallbladder",
        "tool",
    ),
    "palette": [
        (0, 0, 0),
        (255, 255, 100),
        (102, 178, 255),
        (255, 0, 0),
        (0, 102, 51),
        (51, 255, 103),
        (255, 151, 53),
    ],
}

train_dataloader = dict(
    batch_size=6,
    num_workers=5,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo={"classes": ("object",)},
        data_prefix=dict(img="Endoscapes2023/images/train/"),
        ann_file="Endoscapes2023/annotations/train/annotation_coco_class_agnostic.json",
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img="Endoscapes2023/images/val/"),
        ann_file="Endoscapes2023/annotations/val/annotation_coco.json",
    ),
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img="Endoscapes2023/images/test/"),
        ann_file="Endoscapes2023/annotations/test/annotation_coco.json",
    ),
)

orig_val_evaluator = _base_.val_evaluator
orig_val_evaluator[0]["ann_file"] = (
    "{}/Endoscapes2023/annotations/val/annotation_coco.json".format(data_root)
)
val_evaluator = orig_val_evaluator

orig_test_evaluator = _base_.test_evaluator

orig_test_evaluator[0]["ann_file"] = (
    "{}/Endoscapes2023/annotations/test/annotation_coco.json".format(data_root)
)
orig_test_evaluator[1]["ann_file"] = (
    "{}/Endoscapes2023/annotations/test/annotation_coco.json".format(data_root)
)

test_evaluator = orig_test_evaluator
