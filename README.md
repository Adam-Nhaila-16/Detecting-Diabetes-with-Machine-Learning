# Detecting-Diabetes-with-Machine-Learning

## How to Run the Code

#### To compile the code smoothly and access datasets at the same time, we recommend this approach as it requires no modification in paths and cells are compiled smoothly.

1.  **Prerequisites:**

- A Kaggle account (log in at [https://www.kaggle.com](https://www.kaggle.com)).

  

2.  **Steps to Run:**

- Log in to your Kaggle account.

- Go to the **"Code"** section and create a new notebook.

- Import the provided notebook file.

- All modules and libraries are imported in the cells.

- The dataset is accessible in the right part in Datasets as `datasetkaggle`.

  

#### **Alternative approach to run it locally:**

- Have Python installed on your system, create a virtual environment, and ensure `pip` is installed (we recommend using VS Code).

- Install these libraries by running this command in terminal:

```bash

pip install numpy matplotlib pandas scikit-learn lightgbm xgboost catboost tqdm seaborn imbalanced-learn optuna

```

- Adjust the paths for the dataset provided in the file.

  
  

## Table of Contents

Overview

1. Data

2. Problem Definition & Evaluation Metric

3. Exploratory Data Analysis and Data Preprocessing:

- 3.1 Dataset Overview

- 3.2 Data Distribution and Outliers

- 3.3 Features Transformation

- 3.4 Feature Engineering

- 3.5 Feature Selection

- 3.6 Data Balancing and Augmentation Strategies

4. Model Training and Hyperparameter Tuning:

- 4.1 Impact of Training Sampling Techniques

- 4.2 Impact of Scalers

- 4.3 Hyperparameter Tuning

5. Evaluation of all the Models with the Best Hyperparameters:

6. Model Optimization and Ensembling Techniques:

- 6.1 Majority Voting (Hard, Soft and Weighted Soft)

- 6.2 Stacking Ensembles

- 6.3 Evaluation of all the Models on the Validation Set

7. Testing the Final Models

8. Using The Best Model to Predict the Competition Test Set

  
  

## Overview

We predict diabetes using a large dataset of demographic, health, and survey features. Our approach aims to maximise the F1-score ( we noticed class imbalance issue and we provided techniques to face this issues).

  
  

## 1. Data

  

The data provided in the notebook consists of the following csv files:

  

-  **train.csv**: Features of the training set.

-  **labels.csv**: Labels (0 or 1) indicating the diabetes status for the training set.

-  **test.csv**: Test set provided for the competition, without labels.

-  **additional_data_BRFSS2013_2015.csv**: An external dataset derived from the BRFSS (Behavioral Risk Factor Surveillance System) surveys conducted between 2013 and 2015.

  
  

The target variable is **`Diabetes_binary`** (1 = diabetic, 0 = not diabetic).

  
  

## 2. Problem Definition & Evaluation Metric

We classify patients as diabetic or not. The F1-score is the metric chosen to balance precision and recall.

  
  

## 3. Exploratory Data Analysis and Data Preprocessing

  

### 3.1 Dataset Overview

The competition dataset contains 202,944 training samples and 28 features.

  

The target variable, `Diabetes_binary`, is binary, indicating whether a patient has diabetes (Class 1) or not (Class 0). We have :

- 16 binary features

- 7 features with 3-8 unique values

- 4 continuous features
 
- 1 feature with a single unique value

  

### 3.2 Data Distribution and Outliers

- No missing values or duplicate samples.

- Imbalanced class distribution: only 13.97% of samples belong to the positive class (Class 1).

  

### 3.3 Features Transformation

1.  **Categorical Encoding**:

- Features: `BMI Category`, `Age Group`, `Education Level`, `Income Group`

- Encoding method: `OrdinalEncoder` to preserve the natural order of categories.

2.  **Scaling and Normalization**:

- Binary features remain unscaled.

- Continuous features (e.g., `BMI`, `Age`, `GenHlth`) are scaled using `MinMaxScaler` to ensure consistent ranges and improve model performance.

### 3.4 Feature Engineering

We engineered new features by analyzing interactions between `BMI`, `Age`, `GenHlth`, and `Heart Disease Risk`. This resulted in six additional features.

  
  

### 3.5 Feature Selection

- Methods: Recursive Feature Elimination (RFE), model-based importance evaluation, and correlation analysis.

- Decision: Drop the `Age Group` feature (one unique value); retain others due to moderate correlation values (< 0.9).

  

### 3.6 Data Balancing and Augmentation Strategies

We had Class imbalance (Class 1 = 13.97%) so we used these techniques to tackle this problem:

  

1.  **Oversampling**:

- Random Oversampling

- SMOTE

- Borderline-SMOTE

- ADASYN

  

2.  **Undersampling**:

- Random Undersampling

- RENN

  

3.  **External Dataset Integration**:

- Merged cleaned BRFSS 2013 and 2015 datasets to introduce real-world samples.

- Preprocessing steps included data cleaning and aligning feature formats.

## 4. Model Training and Hyperparameter Tuning

  

We train these classifiers:

- Logistic Regression

- Decision Tree

- Random Forest

- Extra Trees

- Balanced Random Forest

- LGBM

- XGBoost

- CatBoost

- AdaBoost

- KNN

- MLP

  

---

  
  

### 4.1 Impact of Training Sampling Techniques

Balancing techniques improved minority class detection.

  

### 4.2 Impact of Scalers

Various scalers were tested; MinMaxScaler provided stable results.

  

### 4.3 Hyperparameter Tuning

We used Grid search and Optuna to refine hyperparameters, it improved F1 significantly.

  

## 5. Evaluation of all the Models with the Best Hyperparameters

  

We evaluated all the models with the best hyperparameters found.

  

## 6. Model Optimization and Ensembling Techniques

  

### 6.1 Majority Voting (Hard, Soft and Weighted Soft)

  
  

### 6.2 Stacking Ensembles

  
  

## 7. Test the Final Models on the Test Set

We test all the models on the test set and pick the best one ( achieving the highest F1 score).

  

## 8. Using The Best Model to Predict the Competition Test Set

We used the best model found (Extra trees) to make our predictions on the competition test set.
