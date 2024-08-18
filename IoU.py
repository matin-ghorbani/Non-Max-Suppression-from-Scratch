import torch


def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    # ...: For every batch
    # boxes preds shape: (N, 4)
    # boxes labels shape: (N, 4)
    # If use only boxes_preds[..., 0] the second dimension will be removed. So we want to keep its shape as (N, 1)

    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == 'corners':
        box1_x1: torch.Tensor = boxes_preds[..., 0:1]
        box1_y1: torch.Tensor = boxes_preds[..., 1:2]
        box1_x2: torch.Tensor = boxes_preds[..., 2:3]
        box1_y2: torch.Tensor = boxes_preds[..., 3:4]

        box2_x1: torch.Tensor = boxes_labels[..., 0:1]
        box2_y1: torch.Tensor = boxes_labels[..., 1:2]
        box2_x2: torch.Tensor = boxes_labels[..., 2:3]
        box2_y2: torch.Tensor = boxes_labels[..., 3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    x2 = torch.min(box1_x2, box2_x2)

    y1 = torch.max(box1_y1, box2_y1)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they don't have intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    union = box1_area + box2_area - intersection + 1e-6
    return intersection / union
