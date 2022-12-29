# Targets : Diagnosis #(M=malignant and B=benign)
# 10 Features: Radius ; Texture; Perimeter; Area; Smoothness...
#... Compactness; Concavity; Concave points; Symmetry; fractal dimension
# Class distribution: 357 Benign and 212 Malignant
# Objective: Classify Breast Cancer into two categories M or B
# DATA PREPROCESSING
# Dataset link: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
# Import Librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

sns.set_theme(style="whitegrid")
data = pd.read_csv("./dataset.csv")
# print(data.head())
# DATA EXPLORATION
print(data.shape) # 569 rows and 33 columns
# data.info()
print(data.select_dtypes(include="object").columns) # Only Diagnosis column contains "Object" values (Categoric values)
print(data.select_dtypes(include=["float64", "int64"]).columns) # Return "float and int" values columns (32) Numerical values
# Statistical summary
print(data.describe())
# Dealing with missing values
print(data.isnull().values.any())
print(data.isnull().values.sum())
print(data.columns[data.isnull().any()]) # Find column wint null values ("Unnamed")
data = data.drop(columns="Unnamed: 32") # New dataset without null values column
# Dealing with categorical data
print(data["diagnosis"].unique())
# Convert categorical values to numerical
data = pd.get_dummies(data=data, drop_first=True)
# print(data.head())
# Countplot
# sns.countplot(x = data["diagnosis_M"])
#plt.show()
print((data.diagnosis_M==0).sum()) # 357 Values M (Class 0)
print((data.diagnosis_M==1).sum()) # 212 Values b (Class 1)
data_2 = data.drop(columns=["diagnosis_M"])
data_2.corrwith(data["diagnosis_M"]).plot.bar(figsize=(20, 10), title="Correlation with diagnosis M", rot=45, grid=True)
# plt.show()
correlation = data.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(correlation, annot=True)
# plt.show()
# SPLITTING OF DATA (TRAIN AND TEST)
X, y = data.iloc[:,1:-1].values, data.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train.shape)
# print(X_test.shape)

# Scaling and Transform
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Build model Logistic regression
classifir_logist = LogisticRegression(random_state=0)
classifir_logist.fit(X_train, y_train)
y_pred = classifir_logist.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

results_metric = pd.DataFrame([["Logistic Regression", acc, f1, prec, rec]], columns=["Model", "Accuracy", "F1 Score", "Precision Score", "Recall"])
# print(results_metric)
confu_mat = confusion_matrix(y_test, y_pred)
print(confu_mat)

# CROSS VALIDATION
accuracies = cross_val_score(estimator=classifir_logist, X = X_train, y=y_train, cv=20)
print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard deviation is {:.2f} %".format(accuracies.std()*100))

# Build Random Forest Classifier model
Classifier_For = RandomForestClassifier(random_state=0)
Classifier_For.fit(X_train, y_train)
y_pred2 = Classifier_For.predict(X_test)
acc_2 = accuracy_score(y_test, y_pred2)
f1_2 = f1_score(y_test, y_pred2)
prec_2 = precision_score(y_test, y_pred2)
rec_2 = recall_score(y_test, y_pred2)
results_forest = pd.DataFrame([["Random Forest ", acc_2, f1_2, prec_2, rec_2]], columns=["Model", "Accuracy", "F1 Score", "Precision Score", "Recall"])
results = results_metric.append(results_forest, ignore_index=True)
print(results)

# CROSS VALIDATION RANDOM FOREST
accuracies_forest = cross_val_score(estimator=Classifier_For, X = X_train, y=y_train, cv=20)
print("Accuracy is {:.2f} %".format(accuracies_forest.mean()*100))
print("Standard deviation is {:.2f} %".format(accuracies_forest.std()*100))