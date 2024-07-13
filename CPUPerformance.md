---

# Computer Hardware Performance Prediction

## Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
%matplotlib inline
```

## Data Visualization and Exploration

### 1. Sanity Check: Print First Two Rows

```python
data = pd.read_csv('path_to_dataset.csv')
print(data.head(2))
```

### 2. Class Imbalance Visualization

```python
data.hist(bins=50, figsize=(30, 30))
plt.show()
```

*Comment on class imbalance observed through the histogram above.*

### 3. Insights into the Dataset

Using Seaborn to visualize the data:

#### Univariate Distribution

```python
sns.displot(data['PRP'], kde=True)
plt.title('Distribution of PRP (Published Relative Performance)')
plt.show()
```

#### Pairplot to Visualize Relationships

```python
sns.pairplot(data)
plt.show()
```

### 4. Correlational Analysis

```python
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

*Discussion on how correlation analysis affects feature selection: Features highly correlated with the target variable (PRP) are considered important and are selected for the model. Features with low correlation might be excluded.*

## Data Pre-processing and Cleaning

### Handling Missing Values

```python
data.isnull().sum()
# Handle missing values if any
data = data.dropna()
```

### Handling Outliers

```python
# Using IQR method to cap outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
```

### Skewness Treatment

```python
data['log_PRP'] = np.log(data['PRP'])
sns.displot(data['log_PRP'], kde=True)
plt.title('Log-Transformed Distribution of PRP')
plt.show()
```

## Model Building

### Splitting the Dataset

```python
# Case 1: 80% Train, 20% Test
X = data.drop(['PRP'], axis=1)
y = data['PRP']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

# Case 2: 10% Train, 90% Test
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.9, random_state=42)
```

*Justification: Different train-test splits are used to observe model performance with varying amounts of training data.*

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5)
model = LinearRegression()
results = cross_val_score(model, X, y, cv=kf, scoring='r2')
print(f'Cross-Validation R² Scores: {results}')
```

### Linear Regression Model Training

```python
# Training on 80% Train, 20% Test Split
model.fit(X_train1, y_train1)
y_pred1 = model.predict(X_test1)
print(f'R²: {r2_score(y_test1, y_pred1)}')
print(f'RMSE: {mean_squared_error(y_test1, y_pred1, squared=False)}')
print(f'MAE: {mean_absolute_error(y_test1, y_pred1)}')

# Training on 10% Train, 90% Test Split
model.fit(X_train2, y_train2)
y_pred2 = model.predict(X_test2)
print(f'R²: {r2_score(y_test2, y_pred2)}')
print(f'RMSE: {mean_squared_error(y_test2, y_pred2, squared=False)}')
print(f'MAE: {mean_absolute_error(y_test2, y_pred2)}')
```

*Explanation: Linear regression is used to predict the target variable `PRP`. Model performance is evaluated using R², RMSE, and MAE to understand the accuracy and error of the predictions.*

---
