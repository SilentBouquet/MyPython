import joblib
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

# 使用GridSearchCV进行参数调优
param_grid = {
    'n_estimators': [100, 200, 300],  # 树的数量
    'max_depth': [None, 10, 20],      # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 每个节点分裂所需的最小样本数
    'class_weight': ['balanced']      # 自动调整类别权重
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 预测
y_pred = best_model.predict(X_test)

# 评估
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred)}")

# 计算Precision-Recall曲线
precision, recall, _ = precision_recall_curve(y_test, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# 保存模型
joblib.dump(best_model, 'RandomForestClassifier_model.pkl')