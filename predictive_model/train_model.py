from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # To save the trained model

# Step 1: Load the dataset
df = pd.read_csv('../input/diabetes.csv')

# Step 2: Prepare features and labels
X = df.drop(columns=['Outcome'])  # Features: All columns except 'Outcome'
y = df['Outcome']                # Target: 'Outcome' column

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Save the model to a file
model_filename = "diabetes_model.pkl"
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# Optional: Load the model to verify it works
loaded_model = joblib.load(model_filename)
sample_input = X_test.iloc[0].values.reshape(1, -1)
sample_prediction = loaded_model.predict(sample_input)
print(f"Sample Prediction: {sample_prediction[0]}")

