import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

# 加载数据
file_path = 'encoded_transactions.xlsx'
data = pd.read_excel(file_path)

# 分离正例和负例
positive_samples = data[data['是否欺诈'] == 1]
negative_samples = data[data['是否欺诈'] == 0]


# 计算特征的重要性
def calculate_feature_importance(X, y):
    model = RandomForestRegressor(n_estimators=20, max_depth=4)
    model.fit(X, y)
    return model.feature_importances_


# 使用 RFE 选择最优特征子集
def select_representative_features(X, y, n_features_to_select):
    feature_importances = calculate_feature_importance(X, y)
    rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    return rfe.get_support()


# 从聚类中选择代表性样本
def select_representative_samples(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    representative_samples = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(clusters == cluster)[0]
        if len(cluster_indices) > 0:
            representative_samples.append(cluster_indices[0])
    return representative_samples


# 选择代表性特征
X_positive = positive_samples.drop(columns=['是否欺诈'])
y_positive = positive_samples['是否欺诈']
selected_features = select_representative_features(X_positive, y_positive, n_features_to_select=10)

# 选择代表性样本
X_negative = negative_samples.drop(columns=['是否欺诈'])
# 确保聚类数量与正例数量相同
n_clusters = len(positive_samples)
representative_negative_indices = select_representative_samples(X_negative, n_clusters=n_clusters)
representative_negative_samples = negative_samples.iloc[representative_negative_indices]

# 合并正例和代表性负例
balanced_data = pd.concat([positive_samples, representative_negative_samples])

# 打乱数据顺序，使正负例交替出现
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存平衡后的数据
balanced_data.to_excel('balanced_transactions.xlsx', index=False)

print(f"原始数据集的类别分布：")
print(data['是否欺诈'].value_counts())
print(f"\n平衡后的数据集的类别分布：")
print(balanced_data['是否欺诈'].value_counts())