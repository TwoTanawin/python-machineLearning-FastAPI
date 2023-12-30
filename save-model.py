import joblib
import pickle

import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier

# Assuming model is an instance of RandomForestClassifier
model = RandomForestClassifier()

# Save the model with the updated scikit-learn version
with open('/app/fast-api/fast-ml/model/rfmodel.pkl', 'wb') as f:
    pickle.dump(model, f)


print("done!")