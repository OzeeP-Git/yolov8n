# Vehicle Detection using YOLOv8n  

## Introduction  
This project implements **YOLOv8n (You Only Look Once – Nano)**, a modern and lightweight object detection model, to detect and classify vehicles in real-world environments. Unlike simple image classification, object detection identifies multiple objects within an image and draws **bounding boxes** around them.  

This work was carried out to demonstrate the effectiveness of YOLOv8n for real-time detection of five vehicle types:  

- Ambulance  
- Bus  
- Car  
- Motorcycle  
- Truck  

---

## Technologies and Tools  

### 1. Python  
The project is implemented in **Python 3.10**, a versatile programming language widely used in machine learning and AI because of its large ecosystem of libraries.  

### 2. Ultralytics YOLOv8  
- Latest version of the YOLO family.  
- Provides pre-trained weights and a user-friendly API.  
- YOLOv8n (Nano) was chosen for its **speed and small size** (~8.9 MB).  
- Supports real-time detection with ~2–3 ms per image on GPU.  

### 3. PyTorch  
- Deep learning framework used as the backend.  
- Provides GPU acceleration and dynamic computation graphs.  
- YOLOv8 is built directly on top of PyTorch.  

### 4. Google Colab  
- Cloud-based Jupyter Notebook environment.  
- Used for GPU training with **Tesla T4 (16 GB VRAM)**.  
- Eliminated the need for high-end local hardware.  

### 5. Local GPU (RTX 3050 Ti)  
- Additional experiments performed locally using **NVIDIA RTX 3050 Ti (4 GB VRAM)**.  
- Enabled small-scale tests without cloud dependency.  

### 6. OpenCV  
- Used for **image preprocessing, frame manipulation, and visualization**.  
- Enabled easy display of bounding boxes on images.  

### 7. NumPy & Matplotlib  
- **NumPy**: Efficient numerical operations.  
- **Matplotlib**: Plots for training curves, evaluation graphs, and confusion matrices.  

---

## Dataset  

### Source  
The dataset was downloaded from **Kaggle** ("Vehicles Detection Dataset").  

### Classes  
- Ambulance  
- Bus  
- Car  
- Motorcycle  
- Truck  

### Features  
- Over 1,200 annotated images.  
- Real-world conditions: day/night, different weather, occlusion, and traffic scenes.  

### Directory Structure  
VehiclesDetectionDataset/
├── images/
│ ├── train/
│ ├── valid/
│ └── test/
├── labels/
│ ├── train/
│ ├── valid/
│ └── test/
└── dataset.yaml

### dataset.yaml  
```yaml
path: /content/VehiclesDetectionDataset

train: images/train
val: images/valid
test: images/test

names: ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
```

### Training YOLOv8n  

#### Command  
```bash
yolo task=detect \
  mode=train \
  model=yolov8n.pt \
  data=/content/VehiclesDetectionDataset/dataset.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  name=YOLOv8-vehicles

```

### Training Configuration
| Parameter      | Value              | Description                          |
| -------------- | ------------------ | ------------------------------------ |
| Model          | yolov8n.pt         | Pre-trained YOLOv8n weights          |
| Epochs         | 50                 | Training cycles                      |
| Batch Size     | 16                 | Images per step                      |
| Image Size     | 640×640            | Input resolution                     |
| Optimizer      | SGD                | With momentum 0.937                  |
| Learning Rate  | 0.01               | Initial LR                           |
| Loss Functions | BCE + CIoU         | Classification + Localization        |
| Augmentations  | Flip, Hue, Scaling | Data augmentation for generalization |

Training duration: ~16 minutes on Google Colab (Tesla T4 GPU).
Results saved in: runs/detect/YOLOv8-vehicles/.



### Model Performance
### Final Metrics
| Class      | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ---------- | --------- | ------ | ------- | ------------ |
| All        | 66.7%     | 57.4%  | 62.2%   | 47.0%        |
| Ambulance  | 81.5%     | 82.5%  | 88.6%   | 75.0%        |
| Bus        | 74.9%     | 69.6%  | 69.4%   | 57.2%        |
| Car        | 64.6%     | 49.6%  | 53.6%   | 37.3%        |
| Motorcycle | 55.3%     | 48.5%  | 57.1%   | 37.0%        |
| Truck      | 57.2%     | 36.7%  | 42.3%   | 28.5%        |

### Observations
- Ambulance and Bus achieved the highest accuracy.
- Car and Truck often confused in complex traffic scenes.
- Motorcycle class underperformed due to fewer training samples.

### Visualizations
- results.png: Loss curves (box, class, DFL), Precision, Recall, mAP.
- confusion_matrix.png: Misclassification between similar classes.
- val_batch.jpg*: Ground truth annotations on validation samples.
- predictions.jpg: Model predictions on test images.
- p_curve.png, pr_curve.png: Precision-Recall analysis.


### Running Inference
### Single Image Prediction
yolo task=detect \
  mode=predict \
  model=runs/detect/YOLOv8-vehicles/weights/best.pt \
  source=/content/sample.jpg \
  conf=0.5

### Video Prediction
yolo task=detect \
  mode=predict \
  model=runs/detect/YOLOv8-vehicles/weights/best.pt \
  source=/content/sample_video.mp4 \
  conf=0.5

### Output predictions will be saved in:
runs/detect/predict/

### Strengths and Limitations
### Strengths

- Extremely fast inference (~2.7 ms/image).
- Compact model suitable for real-time edge deployment.
- Reliable results for Ambulance and Bus classes.

### Limitations

- Struggles with Truck and Motorcycle detection.
- Performance drops under occlusion and low-light conditions.
- Some false positives in crowded environments.

### Applications

- Intelligent traffic surveillance systems.
- Smart city vehicle monitoring.
- Autonomous vehicle perception.
- Emergency vehicle recognition.
- Robotics and IoT edge devices.

### Future Improvements

- Train larger YOLOv8 models (YOLOv8s, YOLOv8m) for improved accuracy.
- Apply advanced augmentations (Mosaic, MixUp, CutMix).
- Use class balancing or synthetic data for Trucks and Motorcycles.
- Fine-tune hyperparameters (confidence thresholds, learning rate).
- Explore knowledge distillation and model ensembles.

### Future Improvements

- Train larger YOLOv8 models (YOLOv8s, YOLOv8m) for improved accuracy.
- Apply advanced augmentations (Mosaic, MixUp, CutMix).
- Use class balancing or synthetic data for Trucks and Motorcycles.
- Fine-tune hyperparameters (confidence thresholds, learning rate).
- Explore knowledge distillation and model ensembles.

### Acknowledgment

This project was developed during my internship at DRDO – Defence Laboratory, Jodhpur. It provided practical experience in applying deep learning-based object detection to real-world vehicle recognition tasks.
