from shapely.geometry import Polygon


def poly2bbox(polygon):
    coords = list(polygon.exterior.coords)
    x1, y1 = coords[0]
    x2, y2 = coords[2]
    return x1, y1, x2, y2


def yolo2bbox(yolo_bbox):
    cx, cy, w, h = yolo_bbox
    
    x1 = cx - (w / 2)
    x2 = cx + (w / 2)
    y1 = cy - (h / 2)
    y2 = cy + (h / 2)
    return [x1, y1, x2, y2]


def bbox2poly(bbox):    
    x1, y1, x2, y2 = bbox

    polygon = Polygon([
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
        (x1, y1)
    ])
    return polygon


def yolo2poly(yolo_bbox):
    bbox = yolo2bbox(yolo_bbox)
    return bbox2poly(bbox)


def tensor2poly(tensor_bbox):
    return bbox2poly(tensor_bbox.tolist()[0])