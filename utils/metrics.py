def compute_iou(box_pred, box_gt):
    x1_pred, y1_pred, x2_pred, y2_pred = box_pred
    x1_gt, y1_gt, x2_gt, y2_gt = box_gt

    # Calculate the Intersection Coordinates
    max_x1 = max(x1_pred, x1_gt)
    max_y1 = max(y1_pred, y1_gt)
    max_x2 = min(x2_pred, x2_gt)
    max_y2 = min(y2_pred, y2_gt)

    # Calculate Intersection Area
    iw = max(0, max_x2 - max_x1 + 1)
    ih = max(0, max_y2 - max_y1 + 1)
    inter_area = iw * ih

    # Calculate Box Areas and Union Area
    box_pred_area = (x2_pred - x1_pred + 1) * (y2_pred - y1_pred + 1)
    box_gt_area = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)
    union_area = box_pred_area + box_gt_area - inter_area

    # Compute the IoU Ratio
    return inter_area / union_area if union_area > 0 else 0