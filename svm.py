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


try:
    dt = pd.read_csv("diabetes.csv")
    print(dt.head())
except FileNotFoundError:
    print("The file does not exist")


# Data Set Information
# The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and
# integrated delivery networks. It includes over 50 features representing patient and hospital outcomes.
# Information was extracted from the database for encounters that satisfied the following criteria:
# (1) It is an inpatient encounter (a hospital admission).
# (2) It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.
# (3) The length of stay was at least 1 day and at most 14 days.
# (4) Laboratory tests were performed during the encounter.
# (5) Medications were administered during the encounter.
# The data contains such attributes as patient number, race, gender, age, admission type, time in hospital,
#  medical specialty of admitting physician, number of lab test performed, HbA1c test result, diagnosis, number of medication,
#   diabetic medications, number of outpatient, inpatient, and emergency visits in the year before the hospitalization, etc.


# 24 features for medications For the generic names: metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide,
# glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, sitagliptin,
# insulin, glyburide-metformin, glipizide-metformin, glimepiride-pioglitazone, metformin-rosiglitazone, and metformin-pioglitazone, the
# feature indicates whether the drug was prescribed or there was a change in the dosage. Values: “up” if the dosage was increased during t
# he encounter, “down” if the dosage was decreased, “steady” if the dosage did not change, and “no” if the drug was not prescribed

# Before diving into modeling:


# Data Preprocessing: Handle missing values, encode

# Feature Selection/Engineering: Identify relevant features that might influence readmission:
                    # We need to think ouit the factors that might influence readmission, factors like sugar (insulin), this is
                    # the hardest and the most challenging
                    # As a result I will use something called XGBOOST which can find the importance features/factors from our
                    # dataset which might affact the readmission

# Model Training: Split  data into training and testing sets, choose appropriate evaluation metrics
 #(like accuracy, precision, recall, or F1-score),and train different models to compare their performance.

# Model Evaluation: Evaluate your models using cross-validation or a separate validation set to ensure they generalize well to new data.

# Model Interpretation: important features driving predictions to gain insights into factors affecting readmission rates.

#data visualiation using historgram

dt.columns

race_counts = dt['readmitted'].value_counts()
print("Unique Races and Their Counts:")
print(race_counts)
race_counts.plot(kind='bar')

print("Unique Races and Their Counts:")
print(race_counts)
race_counts.plot(kind='bar')

race_counts = dt['time_in_hospital'].value_counts().plot(kind='bar')
print("Unique Races and Their Counts:")
print(race_counts)

d1 = dt[dt.gender=='Female'].groupby('age').agg({'number_emergency' : ['sum']})
d1 = pd.DataFrame(d1)
d1.droplevel(0, axis=1)
d1.columns = ['Female']

d2 = dt[dt.gender=='Male'].groupby('age').agg({'number_emergency' : ['sum']})
d2 = pd.DataFrame(d2)
d2.droplevel(0, axis=1)
d2.columns = ['Male']
d1 = d1.join(d2)
d1.plot.barh(figsize=(10,5), legend=True, title="Emergency by gender");
del d1, d2


##Total outpatient by gender
d1 = dt[dt.gender=='Female'].groupby('age').agg({'number_outpatient' : ['sum']})
d1 = pd.DataFrame(d1)
d1.droplevel(0, axis=1)
d1.columns = ['Female']

d2 = dt[dt.gender=='Male'].groupby('age').agg({'number_outpatient' : ['sum']})
d2 = pd.DataFrame(d2)
d2.droplevel(0, axis=1)
d2.columns = ['Male']
d1 = d1.join(d2)
d1.plot.barh(figsize=(10,5), legend=True, title="Total out-patient by gender");
del d1, d2


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'dt' DataFrame is already defined and 'sns' and 'plt' are imported

for i in range(1):
    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(dt['metformin'][dt['diabetesMed']=='Yes'], label='diabetesMed==Yes', ax=ax, color='C1', bins=30)
    sns.histplot(dt['metformin'][dt['diabetesMed']=='No'], label='diabetesMed==No', ax=ax, color='black', bins=30)
    ax.legend()
    ax.grid()

# Show the plot
plt.show()


dt.info()


#data visualization
Steady = dt.loc[dt["insulin"]=="Steady"].count()[0]
Down = dt.loc[dt["insulin"]=="Down"].count()[0]
Up = dt.loc[dt["insulin"]=="Up"].count()[0]


plt.figure(figsize = [5,5], dpi = 120)
labels = ["Steady", "Down", "Up"]

plt.pie([Steady, Down, Up], labels = labels, autopct = "%0.2f%%")
plt.title("Diabetes Patients by Insulin", fontdict = {"fontweight": "bold"})

plt.legend()
plt.show()


#encodeing, changing the string data tpes int int
from sklearn.preprocessing import LabelEncoder
# get only categorical columns list
cat_feats= [col for col in dt.columns if dt[col].dtypes == 'object']

# encode the categorical features
encoder = LabelEncoder()
dt[cat_feats] = dt[cat_feats].apply(encoder.fit_transform)

#spliting the dat int traiinng and testing set

from sklearn.model_selection import train_test_split

y = dt['diabetesMed']  # Target
X = dt.drop(columns = 'diabetesMed')  #Training

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=10, stratify=y)

#standadize the data
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)

#pip install catboost
# install catboost to build the XGBOOST


from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10, shuffle=True)
model = CatBoostClassifier(eval_metric="AUC", task_type="CPU",iterations=10, random_seed=2)

model.fit(X_train, y_train)
pred = model.predict_proba(X_valid)

from sklearn.metrics import accuracy_score
accuracy_score(y_valid, model.predict(X_valid))

from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

xgb_model = XGBClassifier(max_depth=3, tree_method='approx')
xgb_model.fit(X_train, y_train)
xgb = xgb_model.predict_proba(X_valid)

plot_importance(xgb_model)
plt.figure(figsize=(100,100))
pyplot.show()
plt.savefig('important parameters')

#best features
change = dt['change']
insulin = dt['insulin']
glyburide = dt['glyburide']
glipizide = dt['glipizide']
metformin = dt['metformin']
pioglitazone = dt['pioglitazone']
glimepiride = dt['glimepiride']

#building data frame
from pandas import DataFrame
z = DataFrame([change, insulin, glyburide, glipizide, metformin, pioglitazone, glimepiride])
X = z.transpose()
X.columns=['change', 'insulin', 'glyburide', 'glipizide', 'metformin', 'pioglitazone', 'glimeride']
y = y = dt['diabetesMed']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123)


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC


from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'C': [ 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

# fitting the model for grid search
grid.fit(X_train, y_train)


# print best parameter after tuning
print(grid.best_params_)


print(grid.best_estimator_)


# train the model on train set
model =  SVC(C=1.0, break_ties=False, cache_size=200,
                           class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma= 1000, kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False)
model.fit(X_train, y_train)

# print prediction results
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
