import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import joblib

# 1. Load product data
df = pd.read_csv('data/final/products_imputed.csv')

# 2. Binarize target: positive to_buy counts → 1, else 0
df['to_buy_bin'] = (df['to_buy'] > 0).astype(int)

# 3. Prepare features and labels
X = df.drop(columns=['product_id', 'to_buy', 'to_buy_bin'])
X = X.select_dtypes(include=[int, float])  # only numeric columns
y = df['to_buy_bin']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train a simple Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Evaluate
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print(f"Test Accuracy: {accuracy:.3f}")
print(f"Test ROC AUC:  {roc_auc:.3f}")

# 7. Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for To_Buy Prediction')
plt.legend()
plt.tight_layout()
plt.show()

# 8. Save the model
joblib.dump(model, 'models/simple_logreg_to_buy.pkl')

