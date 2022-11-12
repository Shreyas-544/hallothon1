import tensorflow as tf
tf.__version__
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv('/content/drive/MyDrive/Crop_recommendation.csv')

c=data.label.astype('category')
targets = dict(enumerate(c.cat.categories))
data['target']=c.cat.codes
y=data.target
X=data[['N','P','K','temperature','humidity','ph','rainfall']]
y=data['label']

def convert_to_int(x):
    word_dict={'rice':1,'maize':2,'chickpea':3,'kidneybeans':4,'pigeonpeas':5,'mothbeans':6,'mungbean':7,'blackgram':8,'lentil':9,'pomegranate':10,'banana':11,'mango':12,'grapes':13,'watermelon':14,'muskmelon':15,'apple':16,'orange':17,'papaya':18,'coconut':19,'cotton':20,'jute':21,'coffee':22}
    return word_dict[x]
y = y.apply(lambda x : convert_to_int(x))

y = data.iloc[:, -1]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)
k_range = range(1,11)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

from sklearn.svm import SVC
svc_poly = SVC(kernel = 'poly').fit(X_train_scaled, y_train)
print("Poly Kernel Accuracy: ", svc_poly.score(X_test_scaled,y_test))

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

parameters = {'C': np.logspace(-3, 2, 6).tolist(), 'gamma': np.logspace(-3, 2, 6).tolist()}
# 'degree': np.arange(0,5,1).tolist(), 'kernel':['linear','rbf','poly']

model = GridSearchCV(estimator = SVC(kernel="linear"), param_grid=parameters, n_jobs=-1, cv=4)
model.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=4,n_estimators=100,random_state=42).fit(X_train, y_train)

from yellowbrick.classifier import ClassificationReport
classes=list(targets.values())
visualizer = ClassificationReport(clf, classes=classes, support=True,cmap="Blues")

visualizer.fit(X_train, y_train)


from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier().fit(X_train, y_train)
print('Gradient Boosting accuracy : {}'.format(grad.score(X_test,y_test)))

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
prediction_test = model.predict(X_test)    
print(y_test, prediction_test)
print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)

import pickle