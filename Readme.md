# Crop-Weed Detector

This project detects crops and weeds in agricultural images using Ultralytics YOLO26n. Each detected plant is labeled as either `crop` or `weed` with a bounding box.

## Dataset Source

The dataset was obtained from Project AgML on Hugging Face:

[Weed Crop Detection Dataset](https://huggingface.co/datasets/Project-AgML/weed_crop_detection)

The dataset contains 1,120 unique real agricultural field images with approximately 17,700 annotated plant bounding boxes. It was divided into 896 training images, 112 validation images, and 112 testing images.

## Repository contents

- `Main.py` — runs inference and saves annotated images.
- `YOLO_model.py` — downloads/preprocesses the dataset, trains, and evaluates the model.
- `weeds_crops.yaml` — YOLO dataset configuration.
- `input_samples/` — sample input images.
- `predicted/` — sample annotated predictions.
- `runs/detect/` — training and validation results, including trained weights.

The generated `weed_crop_yolo_dataset/` folder is excluded from GitHub because it is approximately 3 GB. It can be recreated by running `YOLO_model.py`.

## Model Performance

- Precision: approximately 77%
- Recall: approximately 76%

Precision represents how many detected plants were classified correctly. Recall represents how many plants present in the image were successfully detected.

## Installation

Install the required libraries:

```bash
pip install ultralytics datasets scikit-learn
```

## Train and evaluate

```bash
python YOLO_model.py
```

The script downloads the [Project AgML weed/crop dataset](https://huggingface.co/datasets/Project-AgML/weed_crop_detection), creates the train/validation/test splits, trains YOLO26n, and evaluates the best checkpoint.

## Run inference

```bash
python Main.py
```

Enter the image path when prompted. The annotated image is saved in `predicted/`, and detected classes with confidence scores are printed in the terminal.
