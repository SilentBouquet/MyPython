import joblib
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, classification_report

# 加载数据
file_path = 'balanced_transactions.xlsx'
data = pd.read_excel(file_path)

# 特征和标签
X = data.drop(columns=['是否欺诈'])
y = data['是否欺诈']
# 确保所有特征都是数值型
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype(str).astype('category').cat.codes

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = EasyEnsembleClassifier(learning_rate=0.01, n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 评估
print(classification_report(y_test, y_pred))
print(f'AUC: {roc_auc_score(y_test, y_pred)}')
print(f'F1: {f1_score(y_test, y_pred)}')
precision, recall, _ = precision_recall_curve(y_test, y_pred)
print(f'PR: {precision}')

# 保存模型
joblib.dump(model, 'RUSBoostClassifier_model.pkl')