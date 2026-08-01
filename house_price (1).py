# House Price Prediction using Linear Regression

# Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset

data = pd.read_csv("F:\AIML Internship\Task 1\housing_price_dataset.csv")

# Step 3: Display Dataset

print("\nFirst 5 Rows of Dataset\n")
print(data.head())

print("\nDataset Information\n")
print(data.info())

print("\nColumn Names\n")
print(data.columns)

# Step 4: Select Required Columns

X = data[['built_up_area', 'bhk', 'bathrooms']]
y = data['price_in_lakhs']

# Step 5: Check Missing Values

print("\nMissing Values\n")
print(data[['built_up_area', 'bhk', 'bathrooms', 'price_in_lakhs']].isnull().sum())

# Remove missing values if any
data = data.dropna(subset=['built_up_area', 'bhk', 'bathrooms', 'price_in_lakhs'])

# Update X and y after removing missing values
X = data[['built_up_area', 'bhk', 'bathrooms']]
y = data['price_in_lakhs']

# Step 6: Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Step 7: Train Model

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# Step 8: Predict

predictions = model.predict(X_test)

# Step 9: Compare Results

comparison = pd.DataFrame({
    "Actual Price (Lakhs)": y_test.values,
    "Predicted Price (Lakhs)": predictions
})

print("\nFirst 10 Predictions\n")
print(comparison.head(10))

# Step 10: Evaluate Model

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("-------------------------")
print("Mean Squared Error :", mse)
print("R2 Score :", r2)

# Step 11: User Prediction

print("\nPredict House Price")

built_up_area = float(input("Enter Built-up Area (sq ft): "))
bhk = int(input("Enter Number of BHK: "))
bathrooms = int(input("Enter Number of Bathrooms: "))

new_house = pd.DataFrame({
    "built_up_area": [built_up_area],
    "bhk": [bhk],
    "bathrooms": [bathrooms]
})

predicted_price = model.predict(new_house)

print("\nEstimated House Price")

print(f"{predicted_price[0]:.2f} Lakhs")

#Predict my own house

new_house = pd.DataFrame({
    "built_up_area": [2000],
    "bhk": [3],
    "bathrooms": [2]
})

price = model.predict(new_house)

print("Predicted Price:", price[0], "Lakhs")