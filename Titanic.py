#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data_cleaned.csv")
print(data.head(5))
print("The shape of dataset is: ", data.shape)

#Segregating variables: Independent and Dependent Variables
x = data.drop(['Survived'], axis=1)
y = data['Survived']
print(x.shape, y.shape)

#Scaling the data (Using MinMax Scaler)
# Importing the MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(x_scaled, columns = x.columns)
print(x)

#Importing the train test split function
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56, stratify=y) #stratify is used for create the same lenght about 0 and 1

#Implementing KNN Classifier
#importing KNN classifier and metric F1score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score

#Creating instance of KNN without set the numbers of neighbors 
clf = KNN()

# Fitting the model
clf.fit(train_x, train_y)

# Predicting over the Train Set and calculating F1
test_predict = clf.predict(test_x)
k = f1_score(test_predict, test_y)
print('Test F1 Score n = 5   ', k )

#Elbow for Classifier
def Elbow(K):
    #initiating empty list
    test_error = []
    #training model for evey value of K
    for i in K:
        #Instance oh KNN
        clf = KNN(n_neighbors = i)
        clf.fit(train_x, train_y)
        # Appending F1 scores to empty list claculated using the predictions
        tmp = clf.predict(test_x)
        tmp = f1_score(tmp,test_y)
        error = 1-tmp
        test_error.append(error)
    return test_error

#Defining K range
k = range(6, 20, 2)

#calling above defined function
test = Elbow(k)

# plotting the Curves
plt.plot(k, test)
plt.xlabel('K Neighbors')
plt.ylabel('Test error')
plt.title('Elbow Curve for test')
plt.show()

#Creating instance of KNN
clf = KNN(n_neighbors = 12)

#Fitting the model
clf.fit(train_x, train_y)

#Predicting over the Train Set and calculating F1
test_predict = clf.predict(test_x)
k = f1_score(test_predict, test_y)
print('Test F1 Score  n = 12  ', k )

print('Code successfully executed!')