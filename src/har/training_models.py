from har.feature_extaction import get_final_df

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import os
import joblib

data = get_final_df()

X_train = data["training_df"].drop("label", axis=1)
y_train = data["training_df"]["label"]

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

X_test = data["testing_df"].drop("label", axis=1)
y_test = data["testing_df"]["label"]
y_test_encoded = le.transform(y_test)

model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
    multi_class='auto'
)

model.fit(X_train, y_train_encoded)

# Ensure models directory exists and save the trained model
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "LogisticRegressionModel.pkl")
joblib.dump(model, model_path)
print(f"Saved trained model to {model_path}")

y_pred = model.predict(X_test)

print("Report of LogisticRegression model: \n")
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
print(confusion_matrix(y_test_encoded, y_pred))

