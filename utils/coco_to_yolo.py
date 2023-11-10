def coco_to_yolo(coco_label):
    """
    Convert coco label to yolo label
    Args:
        coco_label: coco label in the format [x1, y1, x2, y2]
    Returns:
        yolo_label: yolo label in the format [x_center, y_center, width, height]
    """
    x1, y1, x2, y2 = [float(i) for i in coco_label]
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return [x_center, y_center, width, height]