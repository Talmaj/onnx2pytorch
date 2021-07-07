import pytest
import torch

from onnx2pytorch.operations.nonmaxsuppression import NonMaxSuppression


def test_nonmaxsuppression_center_point_box_format():
    op = NonMaxSuppression(center_point_box=1)
    boxes = torch.tensor(
        [
            [
                [0.5, 0.5, 1.0, 1.0],
                [0.5, 0.6, 1.0, 1.0],
                [0.5, 0.4, 1.0, 1.0],
                [0.5, 10.5, 1.0, 1.0],
                [0.5, 10.6, 1.0, 1.0],
                [0.5, 100.5, 1.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], dtype=torch.float32)
    max_output_boxes_per_class = torch.tensor([3], dtype=torch.int64)
    iou_threshold = torch.tensor([0.5], dtype=torch.float32)
    score_threshold = torch.tensor([0.0], dtype=torch.float32)
    exp_selected_indices = torch.tensor(
        [[0, 0, 3], [0, 0, 0], [0, 0, 5]], dtype=torch.int64
    )
    selected_indices = op(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    )
    assert torch.equal(selected_indices, exp_selected_indices)


def test_nonmaxsuppression_flipped_coordinates():
    op = NonMaxSuppression()
    boxes = torch.tensor(
        [
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, 0.9, 1.0, -0.1],
                [0.0, 10.0, 1.0, 11.0],
                [1.0, 10.1, 0.0, 11.1],
                [1.0, 101.0, 0.0, 100.0],
            ]
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], dtype=torch.float32)
    max_output_boxes_per_class = torch.tensor([3], dtype=torch.int64)
    iou_threshold = torch.tensor([0.5], dtype=torch.float32)
    score_threshold = torch.tensor([0.0], dtype=torch.float32)
    exp_selected_indices = torch.tensor(
        [[0, 0, 3], [0, 0, 0], [0, 0, 5]], dtype=torch.int64
    )
    selected_indices = op(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    )
    assert torch.equal(selected_indices, exp_selected_indices)


def test_nonmaxsuppression_identical_boxes():
    op = NonMaxSuppression()
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor(
        [[[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]], dtype=torch.float32
    )
    max_output_boxes_per_class = torch.tensor([3], dtype=torch.int64)
    iou_threshold = torch.tensor([0.5], dtype=torch.float32)
    score_threshold = torch.tensor([0.0], dtype=torch.float32)
    exp_selected_indices = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    selected_indices = op(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    )
    assert torch.equal(selected_indices, exp_selected_indices)


def test_nonmaxsuppression_limit_output_size():
    op = NonMaxSuppression()
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], dtype=torch.float32)
    max_output_boxes_per_class = torch.tensor([2], dtype=torch.int64)
    iou_threshold = torch.tensor([0.5], dtype=torch.float32)
    score_threshold = torch.tensor([0.0], dtype=torch.float32)
    exp_selected_indices = torch.tensor([[0, 0, 3], [0, 0, 0]], dtype=torch.int64)
    selected_indices = op(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    )
    assert torch.equal(selected_indices, exp_selected_indices)


def test_nonmaxsuppression_single_box():
    op = NonMaxSuppression()
    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]], dtype=torch.float32)
    scores = torch.tensor([[[0.9]]], dtype=torch.float32)
    max_output_boxes_per_class = torch.tensor([3], dtype=torch.int64)
    iou_threshold = torch.tensor([0.5], dtype=torch.float32)
    score_threshold = torch.tensor([0.0], dtype=torch.float32)
    exp_selected_indices = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    selected_indices = op(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    )
    assert torch.equal(selected_indices, exp_selected_indices)


def test_nonmaxsuppression_suppress_by_IOU():
    op = NonMaxSuppression()
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], dtype=torch.float32)
    max_output_boxes_per_class = torch.tensor([3], dtype=torch.int64)
    iou_threshold = torch.tensor([0.5], dtype=torch.float32)
    score_threshold = torch.tensor([0.0], dtype=torch.float32)
    exp_selected_indices = torch.tensor(
        [[0, 0, 3], [0, 0, 0], [0, 0, 5]], dtype=torch.int64
    )
    selected_indices = op(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    )
    assert torch.equal(selected_indices, exp_selected_indices)


def test_nonmaxsuppression_suppress_by_IOU_and_scores():
    op = NonMaxSuppression()
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], dtype=torch.float32)
    max_output_boxes_per_class = torch.tensor([3], dtype=torch.int64)
    iou_threshold = torch.tensor([0.5], dtype=torch.float32)
    score_threshold = torch.tensor([0.4], dtype=torch.float32)
    exp_selected_indices = torch.tensor([[0, 0, 3], [0, 0, 0]], dtype=torch.int64)
    selected_indices = op(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    )
    assert torch.equal(selected_indices, exp_selected_indices)


def test_nonmaxsuppression_two_batches():
    op = NonMaxSuppression()
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ],
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor(
        [[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]], [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]],
        dtype=torch.float32,
    )
    max_output_boxes_per_class = torch.tensor([2], dtype=torch.int64)
    iou_threshold = torch.tensor([0.5], dtype=torch.float32)
    score_threshold = torch.tensor([0.0], dtype=torch.float32)
    exp_selected_indices = torch.tensor(
        [[0, 0, 3], [0, 0, 0], [1, 0, 3], [1, 0, 0]], dtype=torch.int64
    )
    selected_indices = op(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    )
    assert torch.equal(selected_indices, exp_selected_indices)


def test_nonmaxsuppression_two_classes():
    op = NonMaxSuppression()
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor(
        [[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3], [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]],
        dtype=torch.float32,
    )
    max_output_boxes_per_class = torch.tensor([2], dtype=torch.int64)
    iou_threshold = torch.tensor([0.5], dtype=torch.float32)
    score_threshold = torch.tensor([0.0], dtype=torch.float32)
    exp_selected_indices = torch.tensor(
        [[0, 0, 3], [0, 0, 0], [0, 1, 3], [0, 1, 0]], dtype=torch.int64
    )
    selected_indices = op(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    )
    assert torch.equal(selected_indices, exp_selected_indices)
