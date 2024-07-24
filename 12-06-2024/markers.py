# -*- coding: utf-8 -*-
"""markers.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AyOmPswKF8eXcwIg7oYkS66DZjRmhPUe
"""

import matplotlib.pyplot as plt


#data
x=[1,2,3,4]
y=[2,4,6,8]

# create a plot with marker
plt.plot(x,y,marker='o',linestyle='--',color='b',markersize=6,markerfacecolor='r')

#add the label as well as title
plt.title("line plot with markers")
plt.xlabel("x-axis")
plt.ylabel("y-axis")

plt.show()