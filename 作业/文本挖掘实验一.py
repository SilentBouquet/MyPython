import os
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def PicManage(path, i):
    pic = Image.open(path)
    pic.c_x, pic.c_y = (int(i / 2) for i in pic.size)
    box = (pic.c_x - 50, pic.c_y - 50, pic.c_x + 50, pic.c_y + 50)
    # 从图片中提取中心100*100的子矩形
    region = pic.crop(box)
    # 切分RGB
    r, g, b = np.split(np.array(region), 3, axis=2)
    # 计算一阶矩
    r_m1 = np.mean(r)
    g_m1 = np.mean(g)
    b_m1 = np.mean(b)
    # 二阶矩
    r_m2 = np.std(r)
    g_m2 = np.std(g)
    b_m2 = np.std(b)
    # 三阶矩
    r_m3 = np.mean(abs(r - r.mean()) ** 3) ** (1 / 3)
    g_m3 = np.mean(abs(g - g.mean()) ** 3) ** (1 / 3)
    b_m3 = np.mean(abs(b - b.mean()) ** 3) ** (1 / 3)
    # 将数据标准化，区间在[-1,1]
    typ = np.array([i])
    arr = np.array([r_m1, g_m1, b_m1, r_m2, g_m2, b_m2, r_m3, g_m3, b_m3])
    # df = pd.DataFrame(preprocessing.minmax_scale(arr,feature_range=(-1,1))).T
    df = pd.DataFrame(arr).T
    dn = pd.DataFrame(typ).T
    return df, dn


result = []
type_result = []

for i in os.listdir('./water/images'):
    if i.endswith('.jpg'):
        df, dn = PicManage('./water/images/' + i, int(i[0]))
        result.append(df)
        type_result.append(dn)

data = pd.concat(result)
typ = pd.concat(type_result)
data = pd.DataFrame(preprocessing.normalize(data, norm='l2'))
data['type'] = typ.values
data.to_csv('picData.csv', index=False)

datapath = './water/moment.csv'
data = pd.read_csv(datapath, encoding='gbk')
data = data.values

# 划分训练集和测试集
# cross_validation在sklearn0.20中改为model_selection
# Start Code Here
# 使用train_test_split函数实现数据集的划分并且改变其数据类型为int
train, test, train_target, test_target = train_test_split(data, type, test_size=0.3, random_state=1, stratify=type)
train = [int(i) for i in train]
# End Code Here ###

# 构建SVM模型
# Start Code Here ###
# 构建SVM模型
model = svm.SVC(kernel='linear', C=1.0, random_state=1)
# End Code Here ###
model.fit(train * 30, train_target)

# save model
joblib.dump(model, 'svcmodel.pkl')

# 混淆矩阵
from sklearn import metrics
# Start Code Here
# 通过调用metrics.confusion_matrix来混淆矩阵
cm_train = metrics.confusion_matrix(train_target, model.predict(train * 30))
cm_test = metrics.confusion_matrix(test_target, model.predict(test * 30))
# End Code Here

train_accuracy = metrics.accuracy_score(train_target, model.predict(train * 30))
test_accuracy = metrics.accuracy_score(test_target, model.predict(test * 30))

print("train accuracy: %f" % train_accuracy)
print("test accuracy: %f" % test_accuracy)

tr = pd.DataFrame(cm_train, index=range(1, 6), columns=range(1, 6)).to_csv('train.csv')
te = pd.DataFrame(cm_test, index=range(1, 6), columns=range(1, 6)).to_csv('test.csv')