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