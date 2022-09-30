# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:A.sharmila 
RegisterNumber:212221230094  
*/
~~~
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
~~~

## Output:
![Screenshot 2022-09-30 225253](https://user-images.githubusercontent.com/94506182/193323874-02a1d4a5-6d5d-4706-a5cd-17360c004583.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
