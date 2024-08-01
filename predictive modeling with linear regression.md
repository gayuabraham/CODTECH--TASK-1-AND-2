import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Assuming the dataset is in a CSV file named 'data.csv'
data = pd.read_csv('/content/student_scores.csv')
print(data)

# Assuming 'Hours' is the independent variable and 'Scores' is the dependent variable
X = data['Hours'].values.reshape(-1, 1)
y = data['Scores'].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Linear Regression model
model = LinearRegression()

# Fitting the model to the training data
model.fit(X_train, y_train)

# Making predictions on the testing data
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Predicting score for a new data point (e.g., 9.25 hours)
new_hours = [[9.25]]
predicted_score = model.predict(new_hours)
print("Predicted Score for 9.25 hours:", predicted_score[0])

import matplotlib.pyplot as plt

# Plotting the regression line
plt.scatter(X_test, y_test, color='blue', label='Actual Scores')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.title('Regression Line: Actual vs Predicted Scores')
plt.legend()
plt.show()

# Plotting actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Scores')
plt.show()
