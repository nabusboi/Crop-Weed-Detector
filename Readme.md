# Crop-Weed Detector using YOLO26n with PyTorch

This project detects crops and weeds in field images using the YOLO26n object-detection model with PyTorch.

## Model Performance

- Precision: approximately 77%
- Recall: approximately 76%

Precision represents how many detected plants were classified correctly. Recall represents how many plants present in the image were successfully detected.

## Running the Backend

Install the required libraries:

```bash
pip install ultralytics datasets scikit-learn
```

To train the model:

```bash
python yolo.py
```

To run the trained model:

```bash
python Main.py
```

Enter the full path of the image when prompted. The detected image will be saved in the `predicted` folder.
