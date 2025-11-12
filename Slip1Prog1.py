# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 1: Create a simple dataset (or load your own CSV) ---
data = {
    'Area': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'Bedrooms': [3, 3, 3, 4, 2, 3, 4, 4, 2, 3],
    'Age': [10, 15, 20, 25, 5, 8, 12, 20, 6, 10],
    'Price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}

df = pd.DataFrame(data)
print("Sample Dataset:")
print(df.head())

# --- Step 2: Divide data into features (X) and target (y) ---
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# --- Step 3: Split data into train and test sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Create and train the model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Step 5: Predict on test data ---
y_pred = model.predict(X_test)

# --- Step 6: Display results ---
print("\nPredicted Prices:", y_pred)
print("Actual Prices:", list(y_test))

# --- Step 7: Evaluate model performance ---
print("\nModel Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
