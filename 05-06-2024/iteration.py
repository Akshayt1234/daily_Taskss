# -*- coding: utf-8 -*-
"""iteration.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ucCxh3OwQfMTLWiUy0okM9VD1VmMiZxZ
"""

import numpy as np
#for in loop is used for iteration in python
#1d array
array_1d=np.array([1,2,3,4,5,6])
#iterate the elements in this array
print("array_1d :",array_1d)

for elements in array_1d:
  print(elements)

#iterating an 2D array
array_2d=np.array([[1,2,3],[4,5,6],[7,8,9]])
print("2D array :",array_2d )

#for rows in array_2d:
  #print(rows)

  #for elements in rows:
    #print(elements)

for elements in np.nditer(array_2d):
  print(elements)

#iterate the elements with index
for element in np.ndeumeatre(array_2d):
  print(f"index:"(index),"Element:" (element))





