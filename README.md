# Computer Hardware Performance Prediction

## Problem Statement

This project involves working with the Computer Hardware Dataset to predict CPU performance. 
The dataset is available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/29/computer+hardware).

## Steps to Complete

### 1. Import Libraries/Dataset
1. **Download the dataset** from the provided link.
2. **Import the required libraries** necessary for data manipulation, visualization, and model building.

### 2. Data Visualization and Exploration
1. **Sanity Check**:
   - Print the first two rows of the dataset to identify all features and ensure the target variable is present.
2. **Class Imbalance**:
   - Use appropriate visualization methods (e.g., bar charts, pie charts) to comment on class imbalance within the dataset.
3. **Data Insight**:
   - Provide various visualizations (e.g., histograms, scatter plots, box plots) to gain insights into the dataset.
4. **Correlational Analysis**:
   - Perform correlational analysis and visualize the results using a heatmap or similar method.
   - Discuss how the correlation analysis will influence feature selection in the next step. Provide justification to earn marks.

### 3. Data Pre-processing and Cleaning 
1. **Pre-processing**:
   - Identify and handle NULL or missing values.
   - Address outliers if present.
   - Manage skewed data appropriately.
   - Mention all pre-processing steps performed in a markdown cell.
   - Explore the latest data balancing techniques and their impact on model evaluation parameters.
2. **Feature Engineering**:
   - Apply necessary feature engineering techniques.
   - Use feature transformation techniques like standardization and normalization based on the dataset’s structure and complexity.
   - Provide proper justification for each technique used. Marks will not be awarded for techniques without justification.
   - Explore and apply methods for identifying feature importance relevant to your feature engineering task.

### 4. Model Building 
1. **Dataset Splitting** :
   - Split the dataset into training and test sets with proper justification:
     - Case 1: Train = 80%, Test = 20%
       - \[ x_train1, y_train1 \] = 80%
       - \[ x_test1, y_test1 \] = 20%
     - Case 2: Train = 10%, Test = 90%
       - \[ x_train2, y_train2 \] = 10%
       - \[ x_test2, y_test2 \] = 90%
2. **Cross-validation** :
   - Explore and apply k-fold cross-validation to ensure model robustness.
3. **Model Training** :
   - Build models using Linear Regression (you can use sklearn or other libraries).
   - Ensure to train and evaluate the models on both train-test splits mentioned.

---
