_base_ = [
    "./S4M_majmin.py",
]

metainfo = {
    "classes": (
        "background",
        "chocolate_cyst",
        "serous_cystadenoma",
        "teratoma",
        "thera_cell_tumor",
        "simple_cyst",
        "normal_ovary",
        "mucinous_cystadenoma",
        "high_grade_serous",
    ),
    "palette": [
        (0, 0, 0),  # background
        (220, 20, 60),  # pupil - crimson red
        (255, 215, 0),  # surgical_tape - golden yellow
        (0, 128, 0),  # hand - green
        (0, 191, 255),  # eye_retractors - deep sky blue
        (138, 43, 226),  # iris - blue violet
        (255, 140, 0),  # skin - dark orange
        (70, 130, 180),  # cornea - steel blue
        (255, 105, 180),  # tool - hot pink
    ],
}


test_dataloader = dict(
    dataset=dict(
        data_root="sample_dataset",
        metainfo=metainfo,
        data_prefix=dict(img="sample_images"),
        ann_file="sample_coco_MMOTU2D.json",
    ),
)


orig_test_evaluator = _base_.test_evaluator

orig_test_evaluator[0]["ann_file"] = "sample_dataset/sample_coco_MMOTU2D.json"
orig_test_evaluator[1]["ann_file"] = "sample_dataset/sample_coco_MMOTU2D.json"

test_evaluator = orig_test_evaluator
