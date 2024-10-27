import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# generate dummy data to simulate real model

# number of rooms
X = np.array([[1], [2], [3], [4], [5]])
# housing price in thousands
y = np.array([100, 150, 200, 250, 300])

# train a linear regression model
model = LinearRegression()
model.fit(X, y)

# save the model to use in the Flask app
joblib.dump(model, 'model.joblib')
