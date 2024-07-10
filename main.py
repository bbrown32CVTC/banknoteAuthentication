# This is a Python script for data cleaning in Machine Learning Unit 1.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create the path to the dataset and set it to a dataframe
path_to_file = "../data_banknote_authentication.csv"
df = pd.read_csv(path_to_file)

# Create a numpy array from the dataframe
banknote_data_array = df.to_numpy()

# Create the X and y arrays using specific columns
X_banknote = banknote_data_array[:, 0:4]
y_banknote = banknote_data_array[:, 4]

# Split the data into 70% training data and 30% test data
X_banknote_train, X_banknote_test, y_banknote_train, y_banknote_test = train_test_split(X_banknote, y_banknote, test_size=0.3)

# Train the scaler, which standarizes all the features to have mean=0 and unit variance.
sc_banknote = StandardScaler()
sc_banknote.fit(X_banknote_train)

# Apply the scaler to the X training data
X_banknote_train_std = sc_banknote.transform(X_banknote_train)

# Apply the same scaler to the X test data
X_banknote_test_std = sc_banknote.transform(X_banknote_test)

# y data is already set to being either 0 or 1 and does not need to be scaled

# Create a perceptron object with the parameters: 40 iterations (epochs) over the training data, and a learning rate of 0.1.
# A random state for the training data is specified to ensure it is shuffled the same way across multiple training runs
pptn_banknote = Perceptron(max_iter=40, eta0=0.1, random_state=0)

# Train the perceptron
pptn_banknote.fit(X_banknote_train_std, y_banknote_train)

# Run the test data through the perceptron and have it predict a y (class label) value for each row.
y_banknote_pred = pptn_banknote.predict(X_banknote_test_std)

# View the predicted y test data (the class that our perceptron put each flower in the testing data into)
print(y_banknote_pred)

# View the true y test data (the actual class label)
print(y_banknote_test)

# View the accuracy of the model, which is: 1 - (observations predicted wrong / total observations) - should be around 80-90%, but since this is a small dataset, there can be a lot of variance depending on exactly how the training and testing data is randomized
print('Accuracy: %.2f' % accuracy_score(y_banknote_test, y_banknote_pred))