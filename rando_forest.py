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
import matplotlib

from sklearn.ensemble import RandomForestClassifier


try:
    dt = pd.read_csv("diabetes.csv")
    print(dt.head())
except FileNotFoundError:
    print("The file does not exist")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search
param_grid = {
    'max_features': [1, 2, 3],
    'min_samples_leaf': [2, 3],
    'n_estimators': [30, 40, 45, 50],
    'min_samples_split': [4, 5, 6],
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 1)


# Fit the grid search to the data
grid_search.fit(X_train, y_train)

grid_search.best_params_

grid_search.best_estimator_

RF = rf.set_params( max_features=1, min_samples_leaf=2, min_samples_split=4,n_estimators=30)
RF.fit(X_train, y_train)
y_pred= RF.predict(X_test)

rf.set_params(max_features=1, min_samples_leaf=2, min_samples_split=4,n_estimators=30)

RF = rf.set_params(max_features=1, min_samples_leaf=2, min_samples_split=4, n_estimators=30)

from sklearn.metrics import classification_report, confusion_matrix
print('Class labels:', np.unique(y))
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(cm)