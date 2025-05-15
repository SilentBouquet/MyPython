import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, classification_report
from sklearn.preprocessing import StandardScaler

# 加载数据
file_path = 'creditcard.csv'
data = pd.read_csv(file_path)

# 特征和标签
X = data.drop(columns=['Class'])
y = data['Class']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 使用SMOTE处理类别不平衡
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    random_state=42,
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train_resampled)

# 预测
y_pred = model.predict(X_test_scaled)

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
joblib.dump(model, 'RandomForestClassifier_model.pkl')