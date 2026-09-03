# PINN for Vehicle State Estimation

## Overview
This repository applies Physics-Informed Neural Networks (PINNs) to estimate dynamic vehicle states, including lateral velocity and sideslip angle. Traditional sensor-based state estimation often suffers from noise, drift, or high sensor hardware costs. By integrating physical vehicle dynamics equations into the loss function, this approach provides robust state estimation from sparse sensor measurements.

## Features
* Physics-constrained neural network loss formulation incorporating vehicle dynamic equations.
* Estimation of unmeasured vehicle state variables under dynamic driving conditions.
* Robust performance against noisy IMU and wheel speed sensor data.
* Validation scripts comparing PINN predictions against numerical vehicle models.

## Technologies
* Programming Language: Python
* Frameworks & Libraries: PyTorch / TensorFlow, NumPy, SciPy, Matplotlib
* Domain Concepts: Vehicle Dynamics, Physics-Informed Machine Learning, State Estimation

## Project Structure
```text
PINN-for-Vehicle-State-Estimation/
├── data/               # Sensor logs and dynamic trajectory data
├── models/             # PINN architecture and loss definitions
├── utils/              # Data pre-processing and vehicle dynamics utilities
├── train.py            # Model training script
└── evaluate.py         # Evaluation and visualization scripts
```

## Results
* Achieved accurate continuous estimation of vehicle sideslip angle and yaw rate.

* Demonstrated lower error metrics compared to purely data-driven baselines in sparse data regimes.

## Future Improvements
* Integration of non-linear tire dynamic models (e.g., Pacejka Magic Formula).

* Real-time online state estimation on edge compute platforms.
