import torch
import bentoml
from bentoml.io import Image, JSON

import glob
import os
import pandas as pd
from PIL import Image
from generate_frames import generate_frames
from tqdm import tqdm


class yolov5runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path="pave_model.pt", force_reload=True
        )

        if torch.cuda.is_available():
            self.model.cuda()
            self.model.amp = False  # Automatic Mixed Precision (AMP) inference
        else:
            self.model.cpu()

        # Config inference settings
        self.inference_size = 640

        # Optional configs
        self.model.conf = 0.05  # NMS confidence threshold
        self.model.iou = 0.2  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.classes = list(range(31))
        self.model.max_det = 1000  # maximum number of detections per image

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, input_imgs):
        # Return predictions only
        results = self.model(input_imgs, size=self.inference_size)
        return results


yolo_v5_runner = bentoml.Runner(yolov5runnable, max_batch_size=30)
svc = bentoml.Service("yolo_v5", runners=[yolo_v5_runner])


def get_length_width(row):
    xmin = row['xmin']
    ymin = row['ymin']
    xmax = row['xmax']
    ymax = row['ymax']

    WIDTH_PIXELS_PER_M = 548.571
    LENGTH_PIXELS_PER_M = 510

    row['width'] = (xmax - xmin) / WIDTH_PIXELS_PER_M
    row['length'] = (ymax - ymin) / LENGTH_PIXELS_PER_M

    return row


@svc.api(input=JSON(), output=JSON())
def invocation(input_dict):
    """
    Takes this data
    data = {
        'filename': "GH051804.MP4",
        'interval': 5 <--- meters interval to capture based on GPS
    }

    where GH051804.MP4 is stored in
        data/
            GH051804.MP4

    It will create a directory
        data/
            GH051804.MP4
            GH051804.bin
            GH051804.gpx
            GH051804/
                GH051804.csv <---- contains prediction in CSV
                frames/
                    0000001.jpg <--- frames extracted from gopro controlled by interval
                predictions/
                    0000001.jpg <--- prediction counterparts

    Returns a JSON dictionary
        {
            'filename': "GH051804.MP4",
            'interval': 5,
            'num_predicted_boxes': 523,         <--- number of predicted boxes
            'num_images': 204,                   <--- number of images extracted
            'csv_location': data/GH051804/GH051804.csv,  <--- local directories in docker, mappable in volume
            'frames': data/GH051804/frames,
            'predictions': data/GH051804/predictions
        }
    """

    # Get info from dictionary
    filename = input_dict["filename"]
    assert filename
    interval = input_dict.get("interval", 5)

    print(f"===========Setup environment {filename}===========")
    # Generate frames first
    basefile_no_ext = os.path.splitext(filename)[0]  # GH051804
    video_filepath = os.path.join("data", filename)  # data/GH051804.MP4
    artifacts_path = os.path.join("data", basefile_no_ext)  # data/GH051804

    # Create artifacts_path
    frames_path = os.path.join(artifacts_path, "frames")
    os.makedirs(frames_path, exist_ok=True)

    predictions_path = os.path.join(artifacts_path, "predictions")
    os.makedirs(predictions_path, exist_ok=True)
    csv_filepath = os.path.join(artifacts_path, f"{basefile_no_ext}.csv")

    # Extract Frames
    print(f"===========Extracting frames for {filename}===========")
    generate_frames(video_filepath, frames_path, interval)

    # Get extracted frames
    files = glob.glob(os.path.join(frames_path, "*.jpg"))
    print(f"===========Starting prediction for {len(files)} images===========")

    predictions_dict = []

    for file in tqdm(files):
        basename = os.path.basename(file)
        batch_ret = yolo_v5_runner.inference.run(file)
        df = batch_ret.pandas().xyxy[0]
        
        df = df.apply(get_length_width, axis=1)
        df["filename"] = basename
        predictions_dict.extend(df.to_dict("records"))

        image = batch_ret.render()[0]
        image = Image.fromarray(image)
        image.save(os.path.join(predictions_path, basename))

    if predictions_dict:
        final_df = pd.DataFrame(predictions_dict)
        final_df.to_csv(csv_filepath)

        return {
            "filename": filename,
            "interval": interval,
            "num_images": len(files),
            "num_predicted_boxes": len(final_df),
            "csv_location": csv_filepath,
            "frames": frames_path,
            "predictions": predictions_path,
        }
    else:
        return {
            "filename": filename,
            "interval": interval,
            "num_images": 0,
            "num_predicted_boxes": 0,
            "csv_location": None,
            "frames": frames_path,
            "predictions": None,
        }
