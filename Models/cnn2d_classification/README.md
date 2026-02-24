# Speckle-Based Flow Rate Estimation (2D Convolutional: Class Based Model)

This project demonstrates how deep learning can be used to estimate microfluidic flow rates (in µL/min) from grayscale speckle pattern videos. It uses a 2D Convolutional Neural Network called BloodFlowCNN2D, which extracts spatial features from each frame. This approach enables fast, non-invasive estimation of flow in microfluidic channels. The project was developed under the guidance of Dr. Christopher Raub as part of a broader study on optical blood flow measurement techniques.

# Background

Speckle patterns occur when coherent light, such as a laser, scatters through dynamic and heterogeneous media like flowing blood. As the fluid's speed or direction changes, the speckle patterns evolve. By analyzing sequences of these patterns, the underlying flow rate can be inferred. This allows for precise, non-invasive measurements of microfluidic flow.

# Project Overview
What This Code Does

Loads short .avi speckle videos named by flow rate, e.g., 5ulpermin.avi.

Splits each video into overlapping grayscale frame stacks.

Trains the BloodFlowCNN2D model to classify flow rates or flow-level categories.

Outputs predictions to CSV files and visualizes results with plots.

Supports evaluation using saved checkpointed model weights.

# Folder Structure
project-root/
│
├── data/                  # Folder for input videos (*.avi)
│
├── models/
│   └── bloodflow_cnn.py   # Core 2D CNN model
│
├── src/
│   ├── dataset.py         # Custom PyTorch Dataset
│   ├── dataloader.py      # Preprocessing and data loader functions
│   └── train.py           # Training pipeline
│
├── outputs/
│   ├── predictions.csv    # CSV of predicted values
│   ├── flowrate_plot.png  # True vs predicted flow rates
│   ├── train_loss.png     # Training loss curves
│   ├── scaler.pkl         # Saved target normalizer
│   └── checkpoints/       # Folder for saved model weights
│
├── main.py                # CLI entry point for training or evaluation
├── requirements.txt       # Python dependencies
└── README.md              # Project overview

# How to Use
1. Install Dependencies

Install required Python packages with:

pip install -r requirements.txt

2. Prepare Your Data

Place all input .avi files in the data/ folder. Expected filenames include:

0ulpermin.avi

5ulpermin.avi

50ulpermin.avi

100ulpermin.avi

150ulpermin.avi

400ulpermin.avi

Each video should have at least 5 frames (default sequence length).

3. Train the Model

Run:

python main.py --mode train


This will:

Train BloodFlowCNN2D for the specified number of epochs.

Save the best model to outputs/checkpoints/.

Generate a CSV of predictions in outputs/predictions.csv.

Plot predicted versus true flow rates in outputs/flowrate_plot.png.

Save training loss curves in outputs/train_loss.png.

4. Evaluate a Trained Model

Run:

python main.py --mode evaluate --checkpoint outputs/checkpoints/best_model.pth


This will reload the model and report performance metrics such as accuracy and a classification report. Optionally, it will produce a confusion matrix to visualize predictions versus true labels. Predictions will be saved in outputs/predictions.csv, including:

True and predicted flow values

Absolute and squared errors

Relative error percentage

Flow rate classification

Example row from predictions:

Index  TrueFlowRate  PredictedFlowRate  AbsoluteError  SquaredError  RelativeError(%)  FlowRateClass
0      400.0         380.42             19.58          383.29        4.9               High

# Requirements

Python packages required include:

torch

torchvision

opencv-python

scikit-learn

matplotlib

pandas

notebook

# Acknowledgements

Developed under the guidance of Dr. Christopher Raub with help from Thuc Pham at The Catholic University of America. This work contributes to the exploration of speckle-based imaging techniques for non-invasive blood flow estimation in microfluidic systems.