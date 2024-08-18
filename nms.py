import torch

from IoU import intersection_over_union

def non_max_suppression(
    predictions,
    prob_thresh,
    iou_thresh,
    box_format='corners',
):
    # Predictions: [[class, probability, x1, y1, x2, y2]]
    assert isinstance(predictions, list)

    bboxes = [
        box
        for box in predictions
        if box[1] >= prob_thresh
    ]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)  # Choose the box with the highest probability

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # If they don't have a same class
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),  # x1, y1, x2, y2
                torch.tensor(box[2:]),  # x1, y1, x2, y2
                box_format=box_format
            ) < iou_thresh
        ]

        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms
