# YOLO KITTI Detector

This project uses YOLO26 detections with a simple tracker.

## Install

```bash
pip install -r requirements.txt
```

## Run

Firstly, download the [KITTI](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) left images, labels and a devkit script to evaluate the model.

Secondly, download the model checkpoint from the [Ultralytics](https://docs.ultralytics.com/models/yolo26) docs.

Here's the project structure.

```text
kitti-yolo-detector/
├── predict.py
├── parse.py
├── requirements.txt
├── tracking/
│   ├── __init__.py
│   ├── bounding_box.py
│   ├── rendering.py
│   ├── track_checking.py
│   ├── track_object.py
│   ├── track_system.py
│   └── utils.py
└── kitti/
    ├── testing/
    │   └── image_02/
    │       └── ****/
    │           └── ******.png
    ├── training/
    │   ├── image_02/
    │   │   └── ****/
    │   │       └── ******.png
    │   └── label_02/
    │       └── ****.txt
    └── devkit_tracking/
        └── devkit/
            └── python/
                ├── data/
                │   └── tracking/
                │       └── label_02/
                │           └── ****.txt
                ├── results/
                │   └── result_sha/
                │       └── data/
                │           └── ****.txt
                └── evaluate_tracking.py
```

Then, run `predict.py`:

```bash
python predict.py \
  --dataset_type training \
  --model_name yolo26s.pt \
  --score_threshold 0.4 \
  --dist_threshold 120.0 \
  --iou_threshold 0.25 \
  --ttl 6 \
  --cyclist_pedestrian_iou 0.0 \
  --begin_index 1 \
  --end_index 1 \
  --no_show_frames
```

Visualize predictions:

```bash
python parse.py \
  --dataset_type training \
  --index 1 \
  --result_sha result_sha \
  --is_gt false
```

# Examples
Screenshots were taken from the first images of 0016 and 0001 training sets.

The 0016 subset.
![Pedestrians](data/screenshot_1.png)

The 0001 subset.
![Cars](data/screenshot_2.png)

# Results
Unfortunately, this model can't be evaluated on the testing KITTI dataset due to its policy. So, only training dataset's results are published here.

|Class|MOTA|MOTP|MT|ML|IDS|FRAG|
|-----|----|----|--|--|---|----|
|Car|58.39%|81.79%|43.97%|9.21%|448|1449|
|Pedestrian|38.17%|71.96%|28.14%|11.97%|269|771|

Good luck!
