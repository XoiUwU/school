import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('posts/Problem Set 6/UniversalBank.csv')  # Replace with your file path
data = data.drop(columns=['ID'])

# ------------------------------
# 1. Partition the data
# ------------------------------
X = data.drop(columns=['Personal Loan'])
y = data['Personal Loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------
# 2. Build Decision Tree
# ------------------------------
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# ------------------------------
# 3. Build Random Forest
# ------------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# ------------------------------
# 4. Build Gradient Boosting Machine
# ------------------------------
gbm = GradientBoostingClassifier(random_state=42)
gbm.fit(X_train, y_train)

# ------------------------------
# 5. Precision and Recall
# ------------------------------
y_pred_dtree = dtree.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_gbm = gbm.predict(X_test)

precision_dtree = precision_score(y_test, y_pred_dtree)
recall_dtree = recall_score(y_test, y_pred_dtree)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
precision_gbm = precision_score(y_test, y_pred_gbm)
recall_gbm = recall_score(y_test, y_pred_gbm)

# ------------------------------
# 6. ROC Curves
# ------------------------------
y_prob_dtree = dtree.predict_proba(X_test)[:,1]
y_prob_rf = rf.predict_proba(X_test)[:,1]
y_prob_gbm = gbm.predict_proba(X_test)[:,1]
fpr_dtree, tpr_dtree, _ = roc_curve(y_test, y_prob_dtree)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_gbm, tpr_gbm, _ = roc_curve(y_test, y_prob_gbm)

# Plot ROC
plt.figure(figsize=(10, 7))
plt.plot(fpr_dtree, tpr_dtree, label=f'Decision Tree (AUC = {roc_auc_score(y_test, y_prob_dtree):.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.2f})')
plt.plot(fpr_gbm, tpr_gbm, label=f'GBM (AUC = {roc_auc_score(y_test, y_prob_gbm):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
