import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression   
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = pd.read_csv("final_data.csv")
data = np.array(data)
data = data[:, 1:]  # Assuming the first column is an index or unnecessary

# Split the data into training and testing sets
msk1 = np.random.rand(len(data)) < 0.7
train = data[msk1]
test = data[~msk1]

# Split features and labels
x_train = train[:, 1:]
y_train = train[:, 0]
x_test = test[:, 1:]
y_test = test[:, 0]

# Scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create and train the logistic regression model
lr = LogisticRegression(max_iter=1000, solver='saga')  # Increased max_iter and changed solver
lr.fit(x_train, y_train)

y_predict=lr.predict(x_test)
print(y_predict)
# Now you can use lr.predict(x_test) to make predictions and evaluate your model
def prediction():
    #learn the regressor
    #print(lr.predict(data))
    count=0
    for i in range(x_test.shape[0]):
        if(lr.predict([x_test[i]])==y_test[i]):
            count+=1
    print("efficiency"+str((count/x_test.shape[0])*100))
def prediction1(data):
    return ((lr.predict(data)))
