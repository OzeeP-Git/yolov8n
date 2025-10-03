#Vehicle Detection using YOLOv8n
Introduction

This project focuses on implementing deep learning-based object detection for the recognition of multiple types of vehicles in outdoor environments. The work is centered around YOLOv8n (You Only Look Once – Nano), a lightweight but powerful object detection architecture designed for real-time deployment.

Object detection is a fundamental task in computer vision. Unlike image classification, which assigns a single label to an entire image, object detection identifies and localizes multiple objects within an image by drawing bounding boxes around them. This capability is highly relevant in domains such as:

Traffic surveillance

Smart city infrastructure

Autonomous driving

Security and defense systems

The objective of this project was to train YOLOv8n on a custom dataset of vehicles to achieve robust detection performance while maintaining efficiency for real-time applications.

Technologies and Tools
1. Python

The entire project is implemented in Python 3.10, which is widely used in machine learning because of its vast ecosystem of scientific computing libraries and frameworks.

2. Ultralytics YOLOv8

The main detection framework used is YOLOv8 by Ultralytics, the latest version of the YOLO family. YOLOv8 introduces anchor-free detection heads, improved feature extraction using C2f and CBS blocks, and support for multiple model scales. We specifically used YOLOv8n (Nano), which is optimized for lightweight deployment with fast inference speed.

Key reasons for using YOLOv8n:

Real-time inference capability (~2–3 ms per image on GPU).

Smaller model size (~8.9 MB), making it suitable for edge devices.

Robust accuracy despite reduced computational cost.

3. PyTorch

PyTorch is the deep learning framework used under the hood by YOLOv8. It provides dynamic computation graphs, GPU acceleration, and a user-friendly API for building, training, and deploying neural networks. YOLOv8 is built on top of PyTorch, enabling flexible customization.

4. Google Colab

Model training was primarily carried out on Google Colab, which provides free access to GPUs such as Tesla T4 (16 GB VRAM). Colab allows running Jupyter Notebooks in the cloud, avoiding the limitations of local hardware.

5. Local GPU (NVIDIA RTX 3050 Ti)

In addition to Colab, training was also performed on a local machine equipped with an NVIDIA GeForce RTX 3050 Ti (4 GB VRAM). Local training enabled experimentation with smaller epochs and hyperparameters without depending on cloud resources.

6. OpenCV

OpenCV (Open Source Computer Vision Library) was used for preprocessing, image loading, and visualization. It provided tools to manipulate frames, resize images, and display detection outputs.

7. NumPy and Matplotlib

NumPy: For numerical operations such as tensor manipulation and dataset processing.

Matplotlib: For plotting training curves (loss, precision, recall, mAP) and generating evaluation visualizations.

Dataset
Source

The dataset was sourced from Kaggle, titled "Vehicles Detection Dataset". It consists of 1250+ annotated images of vehicles.

Classes

The dataset contains five categories:

Ambulance

Bus

Car

Motorcycle

Truck

Scenarios

Images cover diverse environments:

Urban streets and highways

Day and night conditions

Varying weather such as clear, cloudy, and rainy

Occlusion and low-light scenarios

Annotation Format

The dataset was provided in YOLO format, where each .txt file contains normalized bounding box coordinates in the format:

class_id x_center y_center width height

Directory Structure
VehiclesDetectionDataset/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
├── labels/
│   ├── train/
│   ├── valid/
│   └── test/
└── dataset.yaml

dataset.yaml File
path: /content/VehiclesDetectionDataset

train: images/train
val: images/valid
test: images/test

names: ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']

Training YOLOv8n
Training Command
yolo task=detect \
  mode=train \
  model=yolov8n.pt \
  data=/content/VehiclesDetectionDataset/dataset.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  name=YOLOv8-vehicles

Training Configuration
Parameter	Value	Description
Model	yolov8n.pt	Pre-trained YOLOv8n weights
Epochs	50	Number of full dataset passes
Batch Size	16	Number of images per step
Image Size	640×640	Input resolution
Optimizer	SGD	With momentum 0.937
Learning Rate	0.01	Initial LR
Loss Functions	BCE + CIoU	Combines classification and localization losses
Augmentation	Flip, Hue, Scaling	To improve generalization

Training took ~16 minutes on Google Colab with a T4 GPU. Results and logs were stored in:

runs/detect/YOLOv8-vehicles/

Model Performance

Final Metrics
Class	   Precision	Recall	mAP@0.5	 mAP@0.5:0.95
All	        66.7%	   57.4%	 62.2%	 47.0%
Ambulance	  81.5%	   82.5%	 88.6%	 75.0%
Bus	        74.9%	   69.6%	 69.4%	 57.2%
Car	        64.6%	   49.6%	 53.6%	 37.3%
Motorcycle	55.3%	   48.5%	 57.1%	 37.0%
Truck	      57.2%	   36.7%	 42.3%	 28.5%

Observations

Best results: Ambulance and Bus (high precision and recall).

Weaker performance: Trucks and Motorcycles due to occlusion and fewer training examples.

Overall mAP@0.5 = 62.2% demonstrates strong performance for a lightweight model.

Visualizations

results.png: Shows training loss (box, class, DFL), precision, recall, and mAP curves.

confusion_matrix.png: Displays misclassifications (e.g., Car vs Truck confusion).

val_batch.jpg*: Ground truth bounding boxes for validation samples.

predictions.jpg: Model predictions on unseen images.

p_curve.png, pr_curve.png: Precision-Recall behavior across confidence thresholds.

Running Inference
Predict on a Single Image
yolo task=detect \
  mode=predict \
  model=runs/detect/YOLOv8-vehicles/weights/best.pt \
  source=/content/sample.jpg \
  conf=0.5

Predict on a Video
yolo task=detect \
  mode=predict \
  model=runs/detect/YOLOv8-vehicles/weights/best.pt \
  source=/content/sample_video.mp4 \
  conf=0.5


Output detections are saved in:

runs/detect/predict/

Strengths and Limitations
Strengths

Very fast inference (~2.7 ms per image).

Compact model size, deployable on embedded systems.

Good detection accuracy on Ambulance and Bus categories.

Limitations

Lower recall on Trucks and Motorcycles.

Struggles in heavily occluded or low-light conditions.

May produce false positives in cluttered scenes.

Applications

Traffic monitoring and surveillance systems.

Real-time vehicle detection for smart cities.

Autonomous driving support.

Emergency response (detecting ambulances in traffic).

Robotics and IoT-based edge AI solutions.

Future Work

Experiment with larger YOLOv8 variants (YOLOv8s, YOLOv8m) to boost accuracy.

Apply advanced augmentation methods such as Mosaic-9, MixUp, and CutMix.

Perform hyperparameter tuning (confidence thresholds, learning rate schedules).

Balance dataset classes or generate synthetic data to improve detection of Trucks and Motorcycles.

Explore ensemble models or knowledge distillation from stronger detectors.

References

Redmon, J. et al. (2016) – You Only Look Once: Unified, Real-Time Object Detection. CVPR.

Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com

PyTorch Documentation: https://pytorch.org

TorchVision Detection Models: https://pytorch.org/vision/stable/models.html

COCO Dataset Format: https://cocodataset.org/#format-data

OpenCV: https://opencv.org

Acknowledgment

This project was developed during my internship at DRDO – Defence Laboratory, Jodhpur, where I gained practical exposure to object detection techniques and deep learning workflows.
