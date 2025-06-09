import numpy as np
import pandas as pd

# Constants
NUM_RECORDS = 2000
np.random.seed(42)  # For reproducibility

# Weight assignments (based on clinical heuristics)
weights = {
    'pain_level': 0.6,
    'fatigue': 0.3,
    'fever': 0.4,
    'hydration': 0.2,  # Low = 0, Normal = 1, High = 2
    'prior_crises': 0.5
}

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hydration encoding
hydration_map = {'Low': 0, 'Normal': 1, 'High': 2}
hydration_reverse = {v: k for k, v in hydration_map.items()}

# Data generation
scores = []
raw_data = []
for _ in range(NUM_RECORDS):
    pain_level = np.random.randint(0, 11)
    joint_pain = np.random.choice([0, 1])
    fatigue = np.random.choice([0, 1])
    fever = np.random.choice([0, 1])
    hydration_level = np.random.choice([0, 1, 2])  # Encoded
    prior_crises = np.random.choice([0, 1, 2, 3])

    # Calculate risk score
    score = (
        weights['pain_level'] * pain_level +
        weights['fatigue'] * fatigue +
        weights['fever'] * fever +
        weights['hydration'] * hydration_level +
        weights['prior_crises'] * prior_crises
    )
    score = sigmoid(score + np.random.normal(0, 0.1))  # Add small noise
    scores.append(score)
    raw_data.append({
        'PainLevel': pain_level,
        'JointPain': joint_pain,
        'Fatigue': fatigue,
        'Fever': fever,
        'HydrationLevel': hydration_reverse[hydration_level],
        'PriorCrises': prior_crises,
        'Score': score  # temporarily store score
    })

# Use the median as the threshold for balance
threshold = np.median(scores)

data = []
for entry in raw_data:
    crisis_likely = 1 if entry['Score'] > threshold else 0
    entry['CrisisLikely'] = crisis_likely
    del entry['Score']  # remove temporary score
    data.append(entry)

print(f"Generated {len(data)} records.")

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv("data/simulated_scd_data.csv", index=False)
print("Data written to data/simulated_scd_data.csv")
