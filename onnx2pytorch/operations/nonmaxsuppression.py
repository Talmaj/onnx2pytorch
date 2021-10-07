import torch
import torchvision
from torch import nn


class NonMaxSuppression(nn.Module):
    def __init__(self, center_point_box=0):
        self.center_point_box = center_point_box
        super().__init__()

    def forward(
        self,
        boxes,
        scores,
        max_output_boxes_per_class=0,
        iou_threshold=0.0,
        score_threshold=0.0,
    ):
        nms_rs_list = []
        for i in range(boxes.shape[0]):
            for j in range(scores.shape[1]):
                for k in range(boxes.shape[1]):
                    if self.center_point_box == 1:
                        boxes[i][k] = torchvision.ops.box_convert(
                            boxes[i][k], "cxcywh", "xyxy"
                        )
                    else:
                        x1, y1, x2, y2 = boxes[i][k]
                        if x1 < x2 and y1 < y2:
                            continue
                        indices = [0, 1, 2, 3]
                        if x1 > x2:
                            indices = [indices[l] for l in (2, 1, 0, 3)]
                        if y1 > y2:
                            indices = [indices[l] for l in (0, 3, 2, 1)]
                        boxes[i][k] = boxes[i][k].gather(0, torch.tensor(indices))
                mask = scores[i][j] >= score_threshold
                nms_rs = torchvision.ops.nms(
                    boxes[i], scores[i][j], float(iou_threshold)
                )[:max_output_boxes_per_class]
                nms_rs_masked = nms_rs[
                    : mask[nms_rs].nonzero(as_tuple=False).flatten().shape[0]
                ]
                batch_index = torch.full((nms_rs_masked.shape[0], 1), i)
                class_index = torch.full((nms_rs_masked.shape[0], 1), j)
                nms_rs_list.append(
                    torch.cat(
                        (batch_index, class_index, nms_rs_masked.unsqueeze(1)), dim=1
                    )
                )
        return torch.cat(nms_rs_list, dim=0)
