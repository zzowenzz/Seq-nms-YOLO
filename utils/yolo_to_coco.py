
def yolo_to_coco(yolo_label):
    """
    Convert yolo label to coco label
    Args:
        yolo_label: yolo label in the format [x_center, y_center, width, height]
    Returns:
        voc_label: voc label in the format [x1, y1, x2, y2]
    """
    x_center, y_center, width, height = [float(i) for i in yolo_label]
    x1 = x_center - width/2
    y1 = y_center - height/2
    x2 = x_center + width/2
    y2 = y_center + height/2
    return [x1, y1, x2, y2]
