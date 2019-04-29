

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.



digits = load_digits()
print("Image Data Shape" , digits.data.shape)
print("Label Data Shape", digits.target.shape)