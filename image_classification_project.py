import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_mldata
dataset=fetch_mldata('MNIST original')

X=dataset.data
y=dataset.target

some_digit=X[66270]
some_digit_image=some_digit.reshape(28,28) #used to convert into a matrix of 28 by 28 because imshow() function wont work with with single dimensional data

plt.imshow(some_digit_image)#Display an image, i.e. data on a 2D regular raster The first two dimensions (M, N) define the rows and columns of the image.
plt.show()

from sklearn.tree import DecisionTreeClassifier
dtf= DecisionTreeClassifier(max_depth=13)
dtf.fit(X,y)

dtf.score(X,y)
dtf.predict(X[[23,14714,26070,52714,66270],0:784])


