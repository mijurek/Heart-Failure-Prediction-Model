# Heart Failure Prediction Model
Python Jupyter Notebook designed for predicting the likelihood of mortality in patients with heart failure. The solution leverages machine learning techniques, specifically the XGBoost classifier, along with various feature engineering and optimization strategies. The data source is the dataset available on [Keggle](https://www.kaggle.com/code/ecemboluk/heart-attack-prediction-with-classifier-algorithms/input).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Visualizations](#visualizations)

---

## Overview

The project consists of two main components:

1. **Model Creation Script (`creating_xgboost_model.ipynb`)**:
   - Loads and preprocesses the dataset.
   - Implements Recursive Feature Elimination with Cross-Validation (RFECV) and hyperparameter optimization using `RandomizedSearchCV`.
   - Evaluates the model and saves it for deployment.

2. **Prediction Interface (`user_interface.ipynb`)**:
   - Loads the trained model.
   - Provides an interactive interface for users to input patient data.
   - Predicts the probability of death based on the input data.

---

## Features

1. **Data Preprocessing**:
   - Removes low-correlation features.
   - Handles categorical variables.

2. **Feature Selection**:
   - Uses RFECV to select the most important features.
   - Removing 'time' variable preventing data leak

3. **Model Training and Tuning**:
   - XGBoost Classifier with hyperparameter optimization.
   - RandomizedSearchCV for efficient parameter tuning.

4. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1 Score.
   - Confusion Matrix

5. **Visualizations**:
   - Learning curve.
   - Feature importance.
   - SHAP summary plot.
   - Confusion matrix heatmap.

6. **Interactive Prediction**:
   - Simple CLI interface for entering patient details.
   - Outputs mortality probability.
  
---

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/heart-failure-prediction.git
   cd heart-failure-prediction
   ```
   
2. Install the required Python libraries:
   ```
   pip install -r requirements.txt
   ```
   
3. Ensure the dataset (heart_failure_clinical_records_dataset.csv) is available in the working directory.

---

## Usage
1. **Train the Model**
Open the `creating_xgboost_model.ipynb` notebook in Jupyter Notebook or JupyterLab and run all cells to:
   - Preprocess the dataset.
   - Train the XGBoost classifier with feature selection and hyperparameter tuning.
   - Save the trained model as xgb_reg.pkl.

2. **Predict with the Model**
Open the `user_interface.ipynb` notebook in Jupyter Notebook or JupyterLab, and follow the interactive prompts to:
   - Enter patient data.
   - Obtain the probability of death prediction.

---

## Results

**Model Performance**
The trained model achieved the following metrics on the test set:
   - Accuracy: 73%
   - Precision: 70%
   - Recall: 64%
   - F1 Score: 67%

---

## Visualizations

**Learning Curve**<br/><br/>
![Learning Curve](https://raw.githubusercontent.com/mijurek/Heart-Failure-Prediction-Model/refs/heads/main/learning_curve.png)
<br/>

**Feature Importance**<br/><br/>
![Feature Importance](https://raw.githubusercontent.com/mijurek/Heart-Failure-Prediction-Model/refs/heads/main/feature_importane.png)
<br/>

**SHAP Summary Plot**<br/><br/>
![SHAP Summary Plot](https://raw.githubusercontent.com/mijurek/Heart-Failure-Prediction-Model/refs/heads/main/shap.png)
<br/>

**Confusion Matrix**<br/><br/>
![Confusion Matrix](https://raw.githubusercontent.com/mijurek/Heart-Failure-Prediction-Model/refs/heads/main/cm.png)
<br/>

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

