#Read data file, and ploting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from matplotlib import figure
#SVM classifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split# split the data into training, and testing
from sklearn.preprocessing import StandardScaler # standardize
from sklearn.metrics import classification_report

# matrics to test the model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,f1_score
from sklearn.model_selection import cross_val_score
