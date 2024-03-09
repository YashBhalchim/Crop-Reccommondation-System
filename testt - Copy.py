import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset into a pandas DataFrame
dataset = pd.read_csv('crop_ds.csv')

# Split the dataset into features (X) and target variable (y)
X = dataset[[ 'N', 'P', 'K', 'temperature' , 'humidity', 'ph', 'rainfall']]
y = dataset['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model to a file
joblib.dump(rf_classifier, 'crop.pkl')
