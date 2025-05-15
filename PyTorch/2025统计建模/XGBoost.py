import joblib
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from scipy.integrate import trapezoid
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, f1_score, roc_curve

# 加载信用卡诈骗数据
file_path = 'creditcard.csv'
data = pd.read_csv(file_path)

# 特征和标签
X = data.drop(columns=['Class'])
y = data['Class']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 使用SMOTE处理类别不平衡
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# 使用XGBoost模型
model = XGBClassifier(
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    n_estimators=1200,
    max_depth=12,
    learning_rate=0.01,
    random_state=42
)

# 训练模型
eval_set = [(X_test_scaled, y_test)]
model.fit(X_train_scaled, y_train_resampled, eval_set=eval_set, verbose=True)

# 绘制学习曲线
matplotlib.use('Qt5Agg')
results = model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_axis, results['validation_0']['logloss'], label='Log Loss')
ax.legend()
plt.ylabel('Value')
plt.xlabel('Epochs')
plt.title('XGBoost Learning Curve')
plt.show()

# 预测
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# 评估
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC-ROC = {roc_auc_score(y_test, y_proba):.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 绘制PR曲线
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AUC-PR = {trapezoid(precision, recall):.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# 保存模型
joblib.dump(model, 'XGBClassifier_model.pkl')