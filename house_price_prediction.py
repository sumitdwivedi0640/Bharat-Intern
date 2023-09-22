# Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (replace 'house_data.csv' with your dataset file)
data = pd.read_csv('house_data.csv')

# Select the features (independent variables) and the target variable (dependent variable)
X = data[['sqft', 'num_bedrooms', 'num_bathrooms']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model performance metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")

# Plot the regression line for one of the features (e.g., 'sqft')
plt.scatter(X_test['sqft'], y_test, color='blue', label='Actual Prices')
plt.plot(X_test['sqft'], y_pred, color='red', linewidth=2, label='Predicted Prices')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.legend()
plt.title('House Price Prediction')
plt.show()
