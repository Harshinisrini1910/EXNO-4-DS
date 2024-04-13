# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/164806ee-16cf-48e8-a6e6-55a841bf29d7)
data.isnull().sum()
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/ae642534-7633-430f-a7d2-06d8cb830b43)
missing=data[data.isnull().any(axis=1)]
missing
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/37a7e2c7-420f-4106-8c37-3821ba79e708)
data2=data.dropna(axis=0)
data2
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/b671586e-7ba8-492e-83b5-e4bd7362b1bf)
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/af36faef-91c1-4f4e-8599-5f21b6083687)
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/824c34b9-d9e7-4319-94c6-aecb4e0ea01f)
data2
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/049ac19c-6610-42e1-a9d9-87534bd7469c)
new_data=pd.get_dummies(data2, drop_first=True)
new_data
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/f0c55711-a727-4fc0-8c5b-44a6e5f98d7b)
columns_list=list(new_data.columns)
print(columns_list)
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/30ae00f3-c227-43d1-9b97-e0cc0ea0eb72)
features=list(set(columns_list)-set(['SalStat']))
print(features)
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/cfc3070e-5860-45ca-aa08-47452924730b)
y=new_data['SalStat'].values
print(y)
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/7dbe1ab9-ed7d-4737-bd7e-b8b69fd2820d)
x=new_data[features].values
print(x)
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/9b4b8847-279e-4157-b545-98f54b6d76c9)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/355c2921-cd09-4b55-afa6-7406ac213180)
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
![image](https://github.com/Harshinisrini1910/EXNO-4-DS/assets/161415847/3b7a61e3-3ea1-43fb-a002-893b234ea896)
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
# RESULT:
       Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
