import torch

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
yolo = torch.hub.load('ultralytics/yolov5', "yolov5s", pretrained=True)

def get_dmap(img):

    with torch.no_grad():
        preds = midas(transform(img))
        preds = torch.nn.functional.interpolate(
            preds.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    dmap = preds.numpy().copy()
    dmap = dmap - dmap.min()
    dmap = dmap/dmap.max()

    return dmap

def get_object_bboxes(img):
    with torch.no_grad():
        df = yolo(img).pandas().xyxy[0]

    for col in ['xmin', 'ymin', 'xmax', 'ymax']:
                df[col] = df[col].astype(int)

    return df
