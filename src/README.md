# TDNN-Based SGP4 Residual Correction - README

This project implements a Time-Delay Neural Network (TDNN) that learns to
predict orbit residuals (position error) from SGP4 state history, residual history,
and Δ-residual history. The network statistically corrects SGP4 by predicting
the next residual in ECEF meters and adding it to the SGP4-estimated position.

## PROJECT STRUCTURE
project/
|
+-- data/                Input data
+-- requirements.txt     Code Dependencies
+-- main.py              Full TDNN training + testing script
+-- README.txt           This file
+-- models/              Saved models may be stored here
+-- plots/

## MODEL SUMMARY
The TDNN receives a sliding history window (example: 180 samples) where each time
step contains:
- Residual: err_x, err_y, err_z
- Delta Residual: change of residual (difference)
- Normalized SGP4 State:
    x_sgp4, y_sgp4, z_sgp4
    vx_sgp4, vy_sgp4, vz_sgp4 (velocity optional, ON by default)

These features are concatenated and passed through the following network:
1) Conv1D (kernel size 5) learns short-term error patterns
2) Conv1D (kernel size 3, dilation 2) learns longer-term patterns
3) Flatten entire time history into one feature vector
4) Dense MLP layers map temporal features to a 3-D residual prediction

Final corrected position:
Corrected Position = SGP4 Output + Predicted Residual


## INPUT DATA REQUIREMENTS
The /data/ folder must contain one or more files with the following text format:

time  x_sgp4  y_sgp4  z_sgp4  vx_sgp4  vy_sgp4  vz_sgp4  err_x  err_y  err_z

## INSTALLATION
pip install -r requirements.txt

## RUNNING THE MODEL
python main.py

If no saved model exists, a new one will be trained.
If a model already exists, training is skipped and testing is run immediately.

## OUTPUT & EVALUATION
The program displays:
- Training and validation loss curves
- True vs predicted residuals
- SGP4 vs corrected 3D position error
- Histogram comparison of SGP4 and Prediction
