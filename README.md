# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python.
2. Set variables for assigning dataset values.
3. Import LinearRegression from the sklearn
4. APredict the regression for marks by using the representation of graph.
5. ssign the points for representing the graph.
6. Predict the regression for marks by using the representation of graph.
7. Compare the graphs and hence we obtain the LinearRegression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:A.sharmila 
RegisterNumber:212221230094  


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/content/student_scores - student_scores.csv")
df.head()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

*/
```

## Output:
![v1](https://user-images.githubusercontent.com/94506182/203786347-4ee1210e-d1f1-4929-81ae-f43941f6cbb6.png)
![v2](https://user-images.githubusercontent.com/94506182/203786407-c6db0567-d778-4916-97f7-d5f85f67e4fc.png)
![v3](https://user-images.githubusercontent.com/94506182/203786710-39e00f43-7988-471c-b246-594a19783a8c.png)
![v4](https://user-images.githubusercontent.com/94506182/203786656-ae4afbaf-89c8-4357-8290-ced5da7f9ce0.png)
![v5](https://user-images.githubusercontent.com/94506182/203786838-fabcd642-8536-4037-8f91-af900325279f.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
