from sklearn.ensemble import IsolationForest
from joblib import dump
import numpy as np

# Generate sample data
X = np.random.randn(200, 2)  # two features

# Train model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)

# Save model
dump(model, "anomaly-model.joblib")

print("✅ Model trained and saved as anomaly-model.joblib")
