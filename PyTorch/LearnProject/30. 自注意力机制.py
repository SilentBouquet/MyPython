import torch
import torch.nn.functional as F

sentence = torch.tensor([
    0,  # can
    7,  # you
    1,  # help
    2,  # me
    5,  # to
    6,  # translate
    4,  # this
    3   # sentence
])

# 计算自注意力权重
torch.manual_seed(0)
embed = torch.nn.Embedding(10, 16)
# detach()将结果从计算图中分离出来，通常用于评估模式，不进行梯度计算
embedded_sentence = embed(sentence).detach()
print(embedded_sentence.shape)

omega = embedded_sentence.matmul(embedded_sentence.T)
attention_weights = F.softmax(omega, dim=1)
print(attention_weights.shape)
print(attention_weights)
# dim参数用于指定操作的维度。dim=1表示沿着第1个维度（即行）进行求和操作
print(attention_weights.sum(dim=1))

# 计算上下文向量
context_vector = torch.matmul(attention_weights, embedded_sentence)
print(context_vector.shape)
print(context_vector[1])
print()

# 缩放点积注意力
# 初始化查询、键、值向量
torch.manual_seed(0)
d = embedded_sentence.shape[1]
U_query = torch.randn(d, d)
U_key = torch.randn(d, d)
U_value = torch.randn(d, d)

querys = U_query.matmul(embedded_sentence.T).T
keys = U_key.matmul(embedded_sentence.T).T
values = U_value.matmul(embedded_sentence.T).T

query_2 = U_query.matmul(embedded_sentence[1])
print(torch.allclose(querys[1], query_2))

# 计算非归一化的注意力权重
omega_2 = query_2.matmul(keys.T)
print(omega_2)

# 计算归一化的注意力权重
attention_weights_2 = F.softmax(omega_2 / d ** 0.5, dim=0)
print(attention_weights_2)

# 计算上下文嵌入向量
context_vector_2 = attention_weights_2.matmul(values)
print(context_vector_2)