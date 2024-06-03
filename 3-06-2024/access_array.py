# -*- coding: utf-8 -*-
"""access_array.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CHlzxNkrAzy_bleNGe0TYHnMUi2WXc49
"""

import numpy as np
#created an array
array_2d=np.array([[1,2,3],[4,5,6]])
#accsssing an single element
element =array_2d[1,2]
print('element at the index of[1,2] :',element)

#access by row
# : symbol reffers to entire columns
row=array_2d[0:]
print("second row:",row)

#access by row
# : symbol reffers to entire columns
column=array_2d[:,1]
print("second coloumn:",column)

#slicing
#access the subarray with row of 0 and 1,column 1 and 2
slice_array=array_2d[0:2,1:3]
print("subarray:",slice_array)
