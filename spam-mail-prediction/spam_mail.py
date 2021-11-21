import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and Pre-Processing

raw_mail_data = pd.read_csv('./data/mail_data.csv')
# print(raw_mail_data)

# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)) , '')

# print the first five rows from the dataframe
# print(mail_data.head())

# checking the number of rows and columns in the dataframe
# print(mail_data.shape)

# Label spam mail as 0 and not spam mail (ham mail) as 1
# spam -> 0           ham -> 1
mail_data.loc[mail_data['Category'] == 'spam' , 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham' , 'Category'] = 1


# Seprating the data as texts and label
X = mail_data['Message']
Y = mail_data['Category']

# print(X)
# print(Y)

# Splitting the data into training data and test data

X_train , X_test, Y_train, Y_test = train_test_split(X , Y, test_size = 0.2, random_state = 3)

# print(X.shape)
# print(X_train.shape)
# print(X_test.shape)

# Transform the text data to feature vectors(numerical values) that can be used to the logistic regression

feature_extraction = TfidfVectorizer(min_df= 1 , stop_words='english' , lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test as integers (dtype = objest - > dtype = integer)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# print(X_train_features )


#----------------Logistic Regression----------------

model = LogisticRegression()

# trainig the Logistic Regression model with the training data
model.fit(X_train_features , Y_train)

# Evaluating the trained model

# Prediction on trainig data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print("Accuracy on training data : " ,accuracy_on_training_data)

# Prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print("Accuracy on test data : " ,accuracy_on_training_data)

#  Building a predictive system
print()
print("Enter an email content: ")
input_mail_content = input()
input_mail = [input_mail_content]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making predictions
prediction = model.predict(input_data_features)


if prediction[0] == 1 :
    print("It is Ham mail")
else:
    print("It is spam mail")