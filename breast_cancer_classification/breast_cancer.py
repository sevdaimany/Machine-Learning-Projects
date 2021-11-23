import numpy as np
import pandas as pd
from scipy.sparse.construct import rand
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Loading the data from sklearn
bearst_cancer_dataset = sklearn.datasets.load_breast_cancer()
# print(bearst_cancer_dataset)

# loading the data to a pandas dataframe

data_frame = pd.DataFrame(bearst_cancer_dataset.data, columns= bearst_cancer_dataset.feature_names)
print(data_frame.head())

# adding the 'tartget' colummn to rhe data frame
data_frame['label'] = bearst_cancer_dataset.target
print(data_frame.tail())
print(data_frame.shape)
# getting some information about the data
print(data_frame.info())

# checking for missing values
print(data_frame.isnull().sum())

# statistical measures about the data
print(data_frame.describe())

# checking the distribution of target variable - how many 0 and 1 is in the dataset
print(data_frame['label'].value_counts())

# 1 -> benign             
# 0 -> Malignant

# calculate mean for every features group by their label
print(data_frame.groupby('label').mean())

# Seprating features and target
X = data_frame.drop(columns='label' , axis = 1)
Y = data_frame['label']

print(X)
print(Y)


X_train , X_test, Y_train, Y_test = train_test_split(X, Y,test_size= 0.2 , random_state=2)

model = LogisticRegression()

model.fit(X_train, Y_train)


# accuracy on training data
Y_train_prediction = model.predict(X_train)
print("training data accuracy is : " , accuracy_score(Y_train , Y_train_prediction))
 
#  accuracy on test data
Y_test_prediction = model.predict(X_test)
print("test data accuracy is : ", accuracy_score(Y_test, Y_test_prediction))


# Building a predictive system

input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

input_data_as_numpy_array = np.asarray(input_data)
print(input_data_as_numpy_array.shape)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
print(input_data_reshaped.shape)

prediction = model.predict(input_data_reshaped)
print("prediction : " ,prediction)

if(prediction[0] == 0):
    print("The bearst cancer is Malignant")
else:
    print("The bearst cancer is Benign")