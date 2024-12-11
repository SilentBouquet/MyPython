import numpy as np
import pandas as pd
from graphviz import Digraph
from sklearn.cluster import KMeans
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

# combine train and test set
train = pd.read_csv('../titanic/train.csv')
test = pd.read_csv('../titanic/test.csv')
full = pd.concat([train, test], ignore_index=True)
full['Embarked'].fillna('S', inplace=True)
full.Fare.fillna(full[full.Pclass == 3]['Fare'].median(), inplace=True)
full.loc[full.Cabin.notnull(), 'Cabin'] = 1
full.loc[full.Cabin.isnull(), 'Cabin'] = 0
full.loc[full['Sex'] == 'male', 'Sex'] = 1
full.loc[full['Sex'] == 'female', 'Sex'] = 0

full['Title'] = full['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
nn = {'Capt': 'Rareman', 'Col': 'Rareman', 'Don': 'Rareman', 'Dona': 'Rarewoman',
      'Dr': 'Rareman', 'Jonkheer': 'Rareman', 'Lady': 'Rarewoman', 'Major': 'Rareman',
      'Master': 'Master', 'Miss': 'Miss', 'Mlle': 'Rarewoman', 'Mme': 'Rarewoman',
      'Mr': 'Mr', 'Mrs': 'Mrs', 'Ms': 'Rarewoman', 'Rev': 'Mr', 'Sir': 'Rareman',
      'the Countess': 'Rarewoman'}
full.Title = full.Title.map(nn)
full.loc[full.PassengerId == 797, 'Title'] = 'Rarewoman'
full.Age.fillna(999, inplace=True)


def girl(aa):
    if (aa.Age != 999) & (aa.Title == 'Miss') & (aa.Age <= 14):
        return 'Girl'
    elif (aa.Age == 999) & (aa.Title == 'Miss') & (aa.Parch != 0):
        return 'Girl'
    else:
        return aa.Title


full['Title'] = full.apply(girl, axis=1)

Tit = ['Mr', 'Miss', 'Mrs', 'Master', 'Girl', 'Rareman', 'Rarewoman']
for i in Tit:
    full.loc[(full.Age == 999) & (full.Title == i), 'Age'] = full.loc[full.Title == i, 'Age'].median()

full.loc[full['Age'] <= 15, 'Age'] = 0
full.loc[(full['Age'] > 15) & (full['Age'] < 55), 'Age'] = 1
full.loc[full['Age'] >= 55, 'Age'] = 2
full['Pclass'] = full['Pclass'] - 1

Fare = full['Fare'].values
Fare = Fare.reshape(-1, 1)
km = KMeans(n_clusters=3).fit(Fare)  # 将数据集分为2类
Fare = km.fit_predict(Fare)
full['Fare'] = Fare
full['Fare'] = full['Fare'].astype(int)
full['Age'] = full['Age'].astype(int)
full['Cabin'] = full['Cabin'].astype(int)
full['Pclass'] = full['Pclass'].astype(int)
full['Sex'] = full['Sex'].astype(int)
# full['Survived']=full['Survived'].astype(int)


dataset = full.drop(columns=['Embarked', 'Name', 'Parch', 'PassengerId', 'SibSp', 'Ticket', 'Title'])
dataset.dropna(inplace=True)
dataset['Survived'] = dataset['Survived'].astype(int)
# dataset=pd.concat([dataset, pd.DataFrame(columns=['Pri'])])
train = dataset[:800]
test = dataset[800:]

model = BayesianNetwork(
    [('Age', 'Survived'), ('Sex', 'Survived'), ('Fare', 'Pclass'), ('Pclass', 'Survived'), ('Cabin', 'Survived')]
)


def showBN(model, save=False):
    # 传入BayesianModel对象，调用graphviz绘制结构图
    node_attr = dict(
        style='filled',
        shape='box',
        align='left',
        fontsize='12',
        ranksep='0.1',
        height='0.2'
    )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    edges = model.edges()
    for a, b in edges:
        dot.edge(a, b)
    if save:
        dot.view(cleanup=True)
    return dot


showBN(model).render('../titanic/bn.gv', view=False)

model.fit(train, estimator=BayesianEstimator, prior_type="BDeu")  # default equivalent_sample_size=5

# 输出节点信息
print("节点信息为：", ", ".join(model.nodes()))
# 输出依赖关系
print("依赖关系为： ", model.edges())
# 查看每个节点的概率分布
for node in model.nodes():
    if node == 'Survived':
        continue
    print("{}的概率分布为：".format(node), model.get_cpds(node).values)


# 创建推断对象
inference = VariableElimination(model)

# 干预分析：性别
# 干预设置：性别为女性（Sex=0）
intervention_women = {'Sex': 0}
# 计算干预后的查询结果
result_women = inference.query(variables=['Survived'], evidence=intervention_women)
# 获取女性生存的概率
women_survival_prob = result_women.values[1]
print("女性生存概率:", women_survival_prob)

# 干预设置：性别为男性（Sex=1）
intervention_men = {'Sex': 1}
# 计算干预后的查询结果
result_men = inference.query(variables=['Survived'], evidence=intervention_men)
# 获取男性生存的概率
men_survival_prob = result_men.values[1]
print("男性生存概率:", men_survival_prob)

# 干预分析：年龄
# 干预设置：年龄为儿童（Age=0）
intervention_child = {'Age': 0}
# 计算干预后的查询结果
result_child = inference.query(variables=['Survived'], evidence=intervention_child)
# 获取儿童生存的概率
child_survival_prob = result_child.values[1]
print("儿童生存概率:", child_survival_prob)

# 干预设置：年龄为成年人（Age=1）
intervention_adult = {'Age': 1}
# 计算干预后的查询结果
result_adult = inference.query(variables=['Survived'], evidence=intervention_adult)
# 获取成年人生存的概率
adult_survival_prob = result_adult.values[1]
print("成年人生存概率:", adult_survival_prob)

# 干预设置：年龄为老年人（Age=2）
intervention_elderly = {'Age': 2}
# 计算干预后的查询结果
result_elderly = inference.query(variables=['Survived'], evidence=intervention_elderly)
# 获取老年人生存的概率
elderly_survival_prob = result_elderly.values[1]
print("老年人生存概率:", elderly_survival_prob)

# 结果预测及准确率
predict_data = test.drop(columns=['Survived'], axis=1)
y_pred = model.predict(predict_data)
print((np.array(y_pred['Survived'].values == test['Survived'].values)).sum() / len(test))