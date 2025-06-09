import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
import os

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/simulated_scd_data.csv")

# Encode 'HydrationLevel' using OneHotEncoding
hydration_encoder = OneHotEncoder(sparse_output=False)
hydration_encoded = hydration_encoder.fit_transform(df[['HydrationLevel']])
hydration_df = pd.DataFrame(hydration_encoded, columns=hydration_encoder.get_feature_names_out(['HydrationLevel']))

# Combine with main dataset
df_encoded = pd.concat([df.drop(columns=['HydrationLevel']), hydration_df], axis=1)

# Features and target
X = df_encoded.drop(columns=['CrisisLikely'])
y = df_encoded['CrisisLikely']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Means array for C++:", list(scaler.mean_))
print("Stds array for C++:", list(scaler.scale_))

# Export as TFLite
class LogisticModel(tf.Module):
    def __init__(self, weights, bias):
        super().__init__()
        self.weights = tf.constant(weights, dtype=tf.float32)
        self.bias = tf.constant(bias, dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, X.shape[1]], dtype=tf.float32)])
    def __call__(self, x):
        return tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.weights)) + self.bias)

# Convert scikit-learn model to TensorFlow
weights = model.coef_.astype(np.float32)
bias = model.intercept_.astype(np.float32)
module = LogisticModel(weights, bias)

# Save and convert to TFLite
tf.saved_model.save(module, "model/logistic_saved_model")
converter = tf.lite.TFLiteConverter.from_saved_model("model/logistic_saved_model")
tflite_model = converter.convert()

# Save TFLite file
with open("model/aetherflow_model.tflite", "wb") as f:
    f.write(tflite_model)

