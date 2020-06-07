import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

income=pd.read_csv('https://drive.google.com/uc?export=download&id=13HzPLzSDfSwOhvpYOrRSWPMoXzGwYQPx')

#Entropy

X = income[income.columns[:-1]].values
y = income['income'].values
dt = DecisionTreeClassifier(max_depth = 3, min_samples_split=100, min_samples_leaf=50, criterion = 'entropy').fit(X, y)
pred_y = dt.predict(X)
accu = accuracy_score(pred_y, y)
print(accu)

import numpy as np

part_len = [list(y).count('>50K'), list(y).count('<=50K')]
part_line = [[], []]
for i in range(len(y)):
    if y[i] == '>50K':
        part_line[0].append(i)
    else:
        part_line[1].append(i)

for i in range(2):
    part_X = X.copy()
    part_X = np.delete(part_X, part_line[i], axis = 0)
    part_y = y.copy()
    part_y = np.delete(part_y, part_line[i], axis = 0)
    part_pred_y = dt.predict(part_X)
    part_accu = accuracy_score(part_pred_y, part_y)
    print(part_accu)
    
index = list(income.columns)
index.remove('income')
figr = plt.figure(dpi = 600)

tree.plot_tree(dt, feature_names = index, class_names = [">50K", "<=50K"], filled = True)

#gini

X = income[income.columns[:-1]].values
y = income['income'].values
dt = DecisionTreeClassifier(max_depth = 3, min_samples_split=100, min_samples_leaf=50, criterion = 'gini').fit(X, y)
pred_y = dt.predict(X)
accu = accuracy_score(pred_y, y)
print(accu)

import numpy as np

part_len = [list(y).count('>50K'), list(y).count('<=50K')]
part_line = [[], []]
for i in range(len(y)):
    if y[i] == '>50K':
        part_line[0].append(i)
    else:
        part_line[1].append(i)

for i in range(2):
    part_X = X.copy()
    part_X = np.delete(part_X, part_line[i], axis = 0)
    part_y = y.copy()
    part_y = np.delete(part_y, part_line[i], axis = 0)
    part_pred_y = dt.predict(part_X)
    part_accu = accuracy_score(part_pred_y, part_y)
    print(part_accu)
    
index = list(income.columns)
index.remove('income')
figr = plt.figure(dpi = 600)

tree.plot_tree(dt, feature_names = index, class_names = [">50K", "<=50K"], filled = True)

