# Speckle Blood Flow Estimation: Comparative Deep Learning Study

This repository presents a comprehensive study of multiple deep learning architectures for estimating microfluidic flow rates (µL/min) from grayscale speckle pattern videos. The project was developed under the guidance of **Dr. Christopher Raub** at **The Catholic University of America** and explores non-invasive optical methods for measuring flow dynamics in microfluidic and biological systems.

---

## Background

When coherent light (e.g., laser) scatters through dynamic, heterogeneous media like flowing blood, speckle patterns form and evolve according to the underlying flow. By analyzing sequences of these speckle frames, it is possible to infer flow rates with high accuracy using machine learning.

This repository compares **six different model architectures**, highlighting their strengths in **regression** (continuous flow estimation) and **classification** (discrete flow states).

---

## Comparative Model Overview

| Model  | Architecture                    | Task       | Notes                                                                 |
|--------|---------------------------------|-----------|-----------------------------------------------------------------------|
| Model 1 | 3D CNN End-to-End               | Regression | Strong temporal encoding; best overall quantitative performance at high dataset coverage |
| Model 2 | 3D CNN                          | Classification | Rapid convergence; high accuracy even with limited data             |
| Model 3 | 2D CNN                          | Regression | Excellent spatial feature extraction; slightly more sensitive at low flows |
| Model 4 | 2D CNN                          | Classification | Perfect accuracy at moderate dataset coverage; fast convergence   |
| Model 5 | 2D CNN + Linear Regression Head | Regression | Efficient mapping from spatial features to flow; robust across medium/high flows |
| Model 6 | 3D CNN + Linear Regression Head | Regression | Combines temporal and linear mapping; slightly below full 3D end-to-end performance |

---

## Repository Structure


speckle-blood-flow-estimation/
├── models/
│ ├── cnn3d_classification/
│ │ ├── train.py
│ │ ├── dataset.py
│ │ └── model.py
│ ├── cnn3d_regression/
│ ├── cnn2d_regression/
│ ├── cnn2d_classification/
│ ├── cnn2d_linear_head/
│ └── cnn3d_linear_head/
├── src/ # Shared utilities (dataset loaders, preprocessing, helpers)
├── data/ # Input .avi speckle videos
├── results/ # Metrics and output CSVs
├── figures/ # Plots and visualizations
├── experiments/ # Logs or experiment configs per model
├── README.md
└── requirements.txt


Each model folder contains its own `train.py`, dataset scripts, and model definitions. Shared code is in `src/` to avoid duplication.

---

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
2. Prepare Data

Place all .avi speckle videos in data/:

0ulpermin.avi
5ulpermin.avi
50ulpermin.avi
100ulpermin.avi
150ulpermin.avi
400ulpermin.avi

Each video should be at least 5 frames long (default sequence length).

3. Train a Model

Navigate to the desired model folder and run:

python train.py

This will:

Train the model for the defined number of epochs

Save best model weights to outputs/checkpoints/

Generate predictions CSV in outputs/predictions.csv

Plot predicted vs true flow rates in figures/

4. Evaluate a Trained Model
python train.py --mode evaluate --checkpoint outputs/checkpoints/best_model.pt

Reports metrics such as:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

R² Score

Outputs predictions including true/predicted flows, absolute and relative errors, and flow classes (for classification models).

Results and Comparative Analysis

Performance across models varies depending on architecture and dataset size. Key observations:

Regression vs Classification: Regression achieves precise continuous flow estimates; classification excels at categorical flow detection with smaller datasets.

2D vs 3D architectures: 3D models capture temporal dynamics; 2D models rely on spatial features and often match performance at higher dataset coverage.

Low flow sensitivity: Residual error is higher for low flow rates in regression models; classification models show minor confusion between adjacent low-flow classes.

A detailed comparison of metrics for each model is available in results/metrics_summary.csv.

Requirements

Python 3.10+

torch

torchvision

opencv-python

scikit-learn

matplotlib

pandas

notebook

Acknowledgements

Developed under the guidance of Dr. Christopher Raub, with contributions from Thuc Pham and Sara Alqasire at The Catholic University of America. This project investigates speckle-based imaging techniques for non-invasive blood flow estimation in microfluidic systems.
