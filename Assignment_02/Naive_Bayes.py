import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# data load
magic=pd.read_csv('https://drive.google.com/uc?export=download&id=1AoCh22pmLHhdQtYdYUAJJqOCwF9obgVO', sep='\t')

X=magic[magic.columns[:-1]].values
y=magic['class'].values

# data partition
trainX,testX,trainY,testY=train_test_split(X,y,stratify=y,test_size=0.2,random_state=11)


#(1) Complete the following user-defined function for Bernoulli naive Bayes.

def BNB(X,y,alpha=1):
    ######## BERNOULLI NAIVE BAYES ########
    # INPUT 
    # X: n by p array (n=# of observations, p=# of input variables)
    # y: output (len(y)=n, categorical variable)
    # alpha: smoothing paramater
    # OUTPUT
    # pmatrix: 2-D array(list) of size c by p with the probability p_ij=P(x_j=1|y_j=i) (i=class, j=feature) 
    #         where c is number of unique classes in y
        
    # TODO: Bernoulli NB
    
    n, p = X.shape
    eleY = np.unique(y)
    c = len(eleY)
    
    avg = []
    for i in range(p):
        avg.append(np.mean(X[:, i]))
    
    countX = np.zeros((c, p)) # for N(x = 1 | y = 'g'), N(x = 1 | y = 'h') == N_tic
    
    for i in range(n):
        for j in range(p):
            if X[i, j] > avg[j]:
                if y[i] == 'g':
                    countX[0, j] += 1
                else:
                    countX[1, j] += 1

    probX = np.zeros((c, p)) # P(x = 1 | y = 'g'), P(x = 1 | y = 'h')


    binaryY = np.zeros((1, p)) # array(1, p) which contains 0 or 1 for BNB based on Y
    countY = [0, 0] # For N('g'), N('h') == N_c
    probY = [0, 0] # array for containing probability for each columns based on Y: P(y_j = i)

    countY[0] = list(y).count('g')
    countY[1] = list(y).count('h')

    for i in range(c):
        for j in range(p):
            probX[i, j] = (countX[i, j] + alpha) / (countY[i] + (alpha * c))
            
    pmatrix = probX
    
    return pmatrix


#(2) First, you have to binarize training set (trainX) of MAGIC Gamma Telescope data set. Each column is converted to binary variable based on the average value. If a value is greater than average, set a value as 1. Otherwise, set a value as 0. Then, using new binarized dataset, calculate  𝑝𝑖𝑗  (i=class,j=feature) with alpha=1.

avg = []
for i in range(trainX.shape[1]):
    avg.append(np.mean(trainX[:, i]))

binaryX = (trainX > trainX.mean(axis = 0)) ** 1

from sklearn.naive_bayes import BernoulliNB
model_bern = BernoulliNB(alpha = 1).fit(binaryX, trainY)

# (3) Based on the calculated p_ij, calculate probability of class g for each test sample (testX) and calculate accuracy for testX with varying cutoff (To binarize testX, use the mean of trainX). Prior probabilities of classes are proportional to ratios of classes in training set. cutoff ∈{0.1,0.15,0.2,0.25,…,0.95}. Draw a line plot (x=cutoff, y=accuracy).

# Probability of class g
prob_2= BNB(binaryX, trainY)
n, p = testX.shape
binaryX = binaryX = (testX > trainX.mean(axis = 0)) ** 1

ptr = list(model_bern.predict_proba(binaryX)[:, 0])

avg = []
for i in range(p):
    avg.append(np.mean(trainX[:, i]))
    
print("BNB:")
print(BNB(binaryX, testY))


prior_g = list(testY).count('g') / n # Prior P(y = 'g')

# Posterior probability

prob_g = [] # array of Posterior probability for each test sample: P(y = 'g' | x = 1) (testX)

for i in range(n): # likelihood: P(x = 1 | y = 'g'), P(x = 1 | y = 'h')
    temp = [1, 1]
    for j in range(p):
        if binaryX[i, j] == 1:
            temp[0] *= prob_2[0][j]
            temp[1] *= prob_2[1][j]
        else:
            temp[0] *= (1 - prob_2[0][j])
            temp[1] *= (1 - prob_2[1][j])
    # P(x = 1 | y = 'g') * P(y = 'g'), P(x = 1 | y = 'h') * P(y = 'h')
    temp[0] *= prior_g
    temp[1] *= (1 - prior_g)
    
    prob_g.append(list(temp / sum(temp)))

prob_g = np.array(prob_g)
prob_g = prob_g[:, 0]



print("P(x = 1 | y = 'g') based on BNB:")
print(prob_2, "\n")

print("P(x = 1 | y = 'g') based on sklearn:")
print(np.exp(model_bern.feature_log_prob_)[0])

print("BNB P(y = 'g'): ", prob_g)
print("sklearn P(y = 'g'): ", ptr)

### Method for calculating accuracy by comparing two array

def score_calc(A, B):
    count = 0
    for i in range(len(B)):
        if A[i] == B[i]:
            count += 1
    return count / len(B)

### Calculating accuracy with varying cutoff: BNB

cutoff = np.arange(0.1, 1, 0.05) # array of cutoff
set_Y = [] # array of estimated y based on prob_g
for i in cutoff:
    cutoff_Y = ['g' if j >= i else 'h' for j in prob_g]
    set_Y.append(cutoff_Y)

acc_Y = [] # array of accuracy based on set_y and testY
for i in range(len(set_Y)):
    acc_Y.append(score_calc(set_Y[i], testY))
    

### Calculating accuracy with varying cutoff

set_Y_sklearn = []
for i in cutoff:
    cutoff_Y = ['g' if j >= i else 'h' for j in ptr]
    set_Y_sklearn.append(cutoff_Y)

### set_Y_sklearn & testY

acc_Y_sklearn_calc = []
for i in range(len(set_Y)):
    acc_Y_sklearn_calc.append(score_calc(set_Y_sklearn[i], testY))
    
acc_Y_sklearn = []
for i in set_Y:
    accuracy_Y = model_bern.score(binaryX, i)
    acc_Y_sklearn.append(accuracy_Y)
    
plt.figure(figsize = (20, 15))
plt.plot(cutoff, acc_Y_sklearn_calc, color = "green", label = "Model_calculated")
plt.plot(cutoff, acc_Y_sklearn, color = "red", label = "Model_score")
plt.plot(cutoff, acc_Y, color = "blue", label = "BNB_calculated")
plt.title("Cutoff ~ Accuracy")
plt.xlabel("cutoff")
plt.ylabel("Accuracy")
plt.xticks(cutoff)
plt.legend(loc = 0, prop = {"size" : 20})
plt.show

plt.figure(figsize = (20, 15))
plt.plot(prob_g, 'x', color = "blue", label = "BNB")
plt.plot(list(model_bern.predict_proba(binaryX)[:, 0]), '+', color = "red", label = "Model")
plt.legend(loc = 0, prop = {"size": 20})
plt.show()

"""
ANSWER
"""

plt.figure(figsize = (10, 5))
plt.plot(cutoff, acc_Y)
plt.title("Cutoff ~ Accuracy")
plt.xlabel("cutoff")
plt.ylabel("Accuracy")
plt.xticks(cutoff)
plt.show