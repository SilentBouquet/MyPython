import kaiwu as kw
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from kaiwu.classical import SimulatedAnnealingOptimizer

kw.license.init(user_id="51648452576182274", sdk_code="AvuHse0ji1LEsAMWjI1Y3u7igbvxxF")

# 定义数据集
Demand_list = [9000, 9400, 9594, 9859, 9958, 10043, 10309, 10512, 10588]

# 定义精确度
k = 6
# 定义c的阈值
c_min = 8000
c_max = 12000
# 计算c的间隔
delta_c = (c_max - c_min) / (2 ** k - 1)
# 定义c的表达式
x_c = kw.qubo.ndarray(k, 'x_c', kw.qubo.Binary)
coef = np.array([2 ** i for i in range(k)])
c = c_min + delta_c * (x_c @ coef)

# 定义预测函数
Y = c
# 定义Phi的阈值
Phi_min = 0.01
Phi_max = 1
# 计算Phi的间隔
delta_Phi = (Phi_max - Phi_min) / (2 ** k - 1)
# 对于一月到八月的数据，定义每个Phi的表达式
for i in range(len(Demand_list) - 1):
    x_Phi = kw.qubo.ndarray(k, 'x_Phi_{}'.format(i+1), kw.qubo.Binary)
    phi = (Phi_min + delta_Phi * (x_Phi @ coef)) * Demand_list[i]
    Y = Y + phi

# 定义损失函数
Delta = (Demand_list[8] - Y) ** 2

# 求解QUBO模型
q = kw.qubo.make(Delta)
ising = kw.qubo.qubo_model_to_ising_model(q)
qm = kw.qubo.qubo_model_to_qubo_matrix(q)['qubo_matrix']
print("Q矩阵为：\n", qm)
im = kw.qubo.qubo_matrix_to_ising_matrix(qm)[0]
# 得到解向量
sm_min = 0
delta_min = 1000
for i in range(100):
    worker = kw.classical.TabuSearchOptimizer(100, size_limit=1)
    sm = np.array(worker.solve(im))[0]
    var = ising.get_variables()
    sol_dict = kw.qubo.get_sol_dict(sm, var)
    delta = kw.qubo.get_val(q, sol_dict)
    if delta < delta_min:
        delta_min = delta
        sm_min = sm

# 将ising变量转变成qubo变量
for i in range(len(sm_min)):
    if sm_min[i] < 0:
        sm_min[i] = 0

print("误差最小时的解向量：\n", sm_min)
print("最小误差为：\n", delta_min)

# 计算参数项列表
Phi_list = []
cnt = 0
for i in range(len(Demand_list) - 1):
    phi_list = [sm_min[j] for j in range(cnt, cnt+k)]
    phi = Phi_min + delta_Phi * (np.array(phi_list) @ coef)
    Phi_list.append(phi)
    cnt += k

# 计算常数项
c_list = [sm_min[i] for i in range(cnt, cnt+k)]
c = c_min + delta_c * (c_list @ coef)

# 计算最终的第九个月的预测值
y = np.array(Phi_list) @ np.array(Demand_list[:8]).T + c

print('参数项为：\n', Phi_list)
print('常数项为：', c)
print('qubo模型在第九个月的预测值为：', y)

# 计算第十个月的预测值
y_qubo = np.array(Phi_list) @ np.array(Demand_list[1:9]).T + c
print('qubo模型在第十个月的预测值为：', y_qubo)

# 使用AR模型来进行预测
order = 2
model = AutoReg(Demand_list[:8], lags=order)
model_fit = model.fit()
predictions = model_fit.predict(start=len(Demand_list)-1, end=len(Demand_list)-1)[0]
print('AR模型在第九个月的预测值为：', predictions)
model = AutoReg(Demand_list, lags=order)
model_fit = model.fit()
predictions = model_fit.predict(start=len(Demand_list), end=len(Demand_list))[0]
print('AR模型在第十个月的预测值为：', predictions)