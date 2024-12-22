# CFD-ML

**Exploring the Intersection Between Computational Fluid Dynamics and Machine Learning**

![CFD-ML Banner](https://github.com/yourusername/CFD-ML/blob/main/banner.png)

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
  - [Task 2: Diffusion Equation Solver](#task-2-diffusion-equation-solver)
    - [Files](#files)
  - [Task 3: Reynolds Stress Tensor Prediction](#task-3-reynolds-stress-tensor-prediction)
    - [Files](#files-1)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Task 2: Running the Diffusion Simulation](#task-2-running-the-diffusion-simulation)
  - [Task 3: Training and Evaluating Machine Learning Models](#task-3-training-and-evaluating-machine-learning-models)
- [Results](#results)
  - [Task 2: Simulation Snapshots](#task-2-simulation-snapshots)
  - [Task 3: Model Performance](#task-3-model-performance)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

Welcome to the **CFD-ML** repository! This project delves into the synergy between **Computational Fluid Dynamics (CFD)** and **Machine Learning (ML)**. It comprises solutions to tasks from a qualifying assignment, specifically:

1. **Task 2:** Implementing a solver for the Diffusion equation using the Finite Difference Method (FDM).
2. **Task 3:** Developing an XGBoost model to predict Reynolds Stress Tensor discrepancies for Reynolds Averaged Navier Stokes (RANS) modeling.

Through these tasks, the repository showcases how traditional CFD simulations can be enhanced and accelerated using machine learning techniques.

## Repository Structure

```
CFD-ML/
├── Task2/
│   ├── initial_conditions.py
│   ├── simulator.py
│   ├── visualizer.py
│   ├── main.py
│   └── results/
├── Task3/
│   └── RF_XGB_Comparison.ipynb
├── dataset/
│   └── ... (dataset files)
├── README.md
└── requirements.txt
```

### Task 2: Diffusion Equation Solver

**Objective:** Implement a 2D diffusion equation solver using the Finite Difference Method (FDM) and visualize the results.

#### Files

- **`initial_conditions.py`**
  
  Defines classes to set up the initial and boundary conditions for the diffusion simulation. It includes abstract and concrete classes for different geometric regions (rectangular and elliptic) with associated scalar functions.

- **`simulator.py`**
  
  Contains the `Simulator` class responsible for executing the diffusion simulation. It handles the time-stepping using an explicit FDM scheme, applies boundary conditions, and records the simulation history.

- **`visualizer.py`**
  
  Provides the `Visualizer` class to visualize the simulation results. It can generate animations and save snapshot images of the scalar field at various time steps.

- **`main.py`**
  
  The entry point for Task 2. It sets up the simulation parameters, initializes regions and boundary conditions, runs the simulation, and triggers the visualization.

- **`results/`**
  
  Directory where simulation snapshots are saved after running `main.py`.

### Task 3: Reynolds Stress Tensor Prediction

**Objective:** Utilize machine learning models, specifically Random Forest and XGBoost regressors, to predict discrepancies in the Reynolds Stress Tensor for RANS modeling.

#### Files

- **`RF_XGB_Comparison.ipynb`**
  
  An interactive Jupyter Notebook that:
  
  - **Loads and preprocesses data** from the provided dataset.
  - **Trains** both Random Forest and XGBoost models.
  - **Makes predictions** on test data.
  - **Visualizes** the predictions against actual DNS data.
  - **Compares** the performance and execution times of both models.

## Getting Started

Follow these instructions to set up and run the projects locally.

### Prerequisites

- **Python 3.7 or higher**
- **pip** package manager
- **Git** (optional, for cloning the repository)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/CFD-ML.git
   cd CFD-ML
   ```

2. **Create a Virtual Environment (Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, install the necessary packages manually:*

   ```bash
   pip install numpy matplotlib scikit-learn xgboost jupyter
   ```

4. **Download Dataset**

   Ensure that the `dataset/` directory contains all necessary data files required for Task 3. If not included, please obtain the dataset as per the assignment guidelines and place it in the `dataset/` folder.

## Usage

### Task 2: Running the Diffusion Simulation

1. **Navigate to Task2 Directory**

   ```bash
   cd Task2
   ```

2. **Run the Simulation**

   Execute the main script:

   ```bash
   python main.py
   ```

   This will:

   - Initialize the simulation domain with predefined regions and boundary conditions.
   - Run the diffusion simulation over the specified time.
   - Generate and display an animation of the diffusion process.
   - Save snapshot images of the scalar field in the `results/` directory.

3. **View Results**

   - **Animation:** The simulation will display an animation window showing the diffusion over time.
   - **Snapshots:** Check the `results/` folder for saved images of the scalar field at various time steps.

### Task 3: Training and Evaluating Machine Learning Models

1. **Navigate to Task3 Directory**

   ```bash
   cd ../Task3
   ```

2. **Open the Jupyter Notebook**

   Launch Jupyter Notebook:

   ```bash
   jupyter notebook RF_XGB_Comparison.ipynb
   ```

3. **Execute the Notebook**

   - **Data Loading:** The notebook will load training and testing data from the `dataset/` directory.
   - **Model Training:** It will train both Random Forest and XGBoost models, recording their training times.
   - **Predictions:** Generate predictions on the test set.
   - **Visualization:** Plot the Reynolds stress anisotropy and compare model predictions against DNS data.
   - **Performance Comparison:** Visualize the execution times of both models.

4. **Review Results**

   - **Plots:** The notebook will display various plots comparing model predictions and actual data.
   - **Execution Times:** A bar chart will illustrate the training times of both models.

## Results

### Task 2: Simulation Snapshots

After running the diffusion simulation, snapshots of the scalar field at different time steps are saved in the `Task2/results/` directory. These images visualize how the scalar quantity diffuses over the domain.

![Snapshot Example](https://github.com/yourusername/CFD-ML/blob/main/Task2/results/snapshot_0020.png)

### Task 3: Model Performance

The Jupyter Notebook `RF_XGB_Comparison.ipynb` presents a comprehensive comparison between Random Forest and XGBoost models in predicting Reynolds Stress Tensor discrepancies.

- **Training Times:**

  | Model          | Training Time (seconds) |
  |----------------|-------------------------|
  | XGBoost        | 0.58                    |
  | Random Forest  | 0.71                    |

  ![Training Times](https://github.com/yourusername/CFD-ML/blob/main/Task3/training_times.png)

- **Prediction Accuracy:**

  Visual comparisons in the Barycentric triangle demonstrate the accuracy of each model in capturing the stress anisotropy.

  ![Model Comparison](https://github.com/yourusername/CFD-ML/blob/main/Task3/model_comparison.png)

## Contributing

Contributions are welcome! If you have suggestions, improvements, or want to report issues, please open an [issue](https://github.com/yourusername/CFD-ML/issues) or submit a pull request.

## License

This project is licensed under the [MIT License](https://github.com/yourusername/CFD-ML/blob/main/LICENSE).

## Contact

For any questions or inquiries, please contact [ahmed.moh.abdelaal@gmail.com](mailto:your.email@example.com).
