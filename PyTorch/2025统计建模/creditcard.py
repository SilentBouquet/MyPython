import joblib
import matplotlib
import mlflow
import requests
import pandas as pd
import mlflow.sklearn
import lightgbm as lgb
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


# 下载数据集
def download_dataset(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"数据集已下载到 {filename}")
    else:
        print("下载失败，请检查URL或网络连接。")


# 数据预处理
def preprocess_data(data_path):
    # 加载数据
    df = pd.read_csv(data_path)
    print("数据集基本信息：")
    print(df.info())
    print("\n数据集描述：")
    print(df.describe())

    # 检查类别分布
    print("\n类别分布：")
    print(df['Class'].value_counts())

    # 分离特征和标签
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# 构建和训练模型
def train_model(X_train, y_train):
    # 使用SMOTE处理类别不平衡
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 构建LightGBM模型
    model = lgb.LGBMClassifier(objective="binary", class_weight='balanced', random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    return model


# 评估模型
def evaluate_model(model, X_test, y_test):
    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 打印分类报告
    print("分类报告：")
    print(classification_report(y_test, y_pred))

    # 计算AUC
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc:.4f}")

    # 绘制ROC曲线
    matplotlib.use('Qt5Agg')
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    return auc


# 保存模型
def save_model(model, scaler, model_path, scaler_path):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"模型已保存到 {model_path}")
    print(f"标准化器已保存到 {scaler_path}")


# 设置MLflow实验
def setup_mlflow(experiment_name):
    mlflow.set_experiment(experiment_name)
    mlflow.autolog(disable=True)


# 记录模型指标
def log_model_metrics(model, X_test, y_test, run_name):
    with mlflow.start_run(run_name=run_name):
        auc = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("AUC", auc)


# 主函数
if __name__ == "__main__":
    # 数据集下载
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    dataset_filename = "creditcard.csv"
    download_dataset(dataset_url, dataset_filename)

    # 数据预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(dataset_filename)

    # 设置MLflow实验
    setup_mlflow("credit_card_fraud_detection")

    # 模型训练
    model = train_model(X_train, y_train)

    # 模型评估
    evaluate_model(model, X_test, y_test)

    # 记录模型指标
    log_model_metrics(model, X_test, y_test, "credit_card_fraud_model")

    # 模型保存
    save_model(model, scaler, "fraud_detection_model.pkl", "scaler.pkl")