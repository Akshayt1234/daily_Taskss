# -*- coding: utf-8 -*-
"""houseprice.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-xwsW_0pV72d-zJjODbd1Oc2fN7_WB2K
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#load our data
df = pd.read_csv('/content/housing.csv')

# split the dataset into feature and target as  (x) and (y) axis
x=df[['size','bedrooms']].values
y=df['price'].values
#initiate or define our model
model=LinearRegression()

#define our crossvalidation methode which is KFold
kf=KFold(n_splits=5)

mae_scores=[]
for train_index, test_index in kf.split(x):
  x_train,x_test=x[train_index],x[test_index]
  y_train,y_test=y[train_index],y[test_index]

  #training the model with sets which gets after looping
  model.fit(x_train,y_train)

  #predict the test set
  y_pred=model.predict(x_test)

  #calculate the mean absolute error
  mae=mean_absolute_error(y_test,y_pred)
  mae_scores.append(mae)

average_mae=np.mean(mae_scores)
print(f"absolute error:{average_mae}")