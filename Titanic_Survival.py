import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
passengers


# Update sex column to numerical
passengers.Sex = passengers.Sex.map({'male':0,'female':1})
passengers

# Fill the nan values in the age column
passengers.Age.fillna(value=passengers.Age.mean(),inplace=True)
passengers['Age']

# Create a first class column
passengers['FirstClass']  = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0) 
passengers[['Pclass', 'Name', 'FirstClass']]

# Create a second class column
passengers['SecondClass']  = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0) 
passengers[['Pclass', 'Name', 'FirstClass', 'SecondClass']]

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers.Survived

# Perform train, test, split
features_train, features_test, labels_train,  labels_test = train_test_split(features,survival,test_size=0.25, random_state=42)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
norm_train_features = scaler.fit_transform(features_train)
norm_test_features = scaler.fit_transform(features_test)

# Create and train the model
model = LogisticRegression()
model.fit(norm_train_features , labels_train)

# Score the model on the train data
model.score(norm_train_features , labels_train)

# Score the model on the test data
model.score(norm_test_features , labels_test)

# Analyze the coefficients
model.coef_

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])

# Sample Passenger features:
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
John_Doe = np.array([1.0,49.0,0.0,0.0])

# Combine passenger arrays
sample_passengers = np.array([Jack , Rose, John_Doe])
sample_passengers = scaler.transform(sample_passengers)
print(model.predict(sample_passengers))


# Scale the sample passenger features
prob = model.predict_proba(sample_passengers)
print(prob)

# Make survival predictions!


prob_df = pd.DataFrame({'Passenger':['Rose', 'Jack', 'John_Doe'], '% Likely To Survive':[ val[0]*100 for val in prob], '% Likely To Not-Survive':[ val[1]*100 for val in prob]})
prob_df


