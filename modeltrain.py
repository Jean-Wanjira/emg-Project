import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the data
data = pd.read_csv('emg_features_with_state_timing1.csv')

# Drop rows with NaN values
data.dropna(inplace=True)

# Extract features and labels
x = data[['MAV', 'RMS', 'Variance', 'Waveform Length', 'SSC', 'IEMG', 'Mean Frequency', 'Median Frequency']]
y = data['State']

# Convert labels to numerical values
y = y.map({'Relaxed': 0, 'Flexed':1})

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalize the feature
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Train a machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
