# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Manikandan M
RegisterNumber:212224040184

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')

print(df.head())
print()

print(df.tail())
print()

x=df.iloc[:,:-1].values
print(x)
print()

y=df.iloc[:,1].values
print(y)
print()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

print(y_pred)
print()

print(y_test)
print()



plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()




plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,regressor.predict(x_test),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()




mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

*/
```

## Output:
<img width="1903" height="983" alt="image" src="https://github.com/user-attachments/assets/a11696bb-0d03-492d-9bf3-8bdddb9b573f" />
<img width="1910" height="981" alt="image" src="https://github.com/user-attachments/assets/cc696e44-1da3-4d70-945f-bb47e38a2f8e" />
<img width="1905" height="984" alt="image" src="https://github.com/user-attachments/assets/3255011a-475f-434a-a710-0ffbb4b7a2c0" />
<img width="1903" height="980" alt="image" src="https://github.com/user-attachments/assets/da41aed6-4de3-43c7-9243-045f5857617d" />





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
