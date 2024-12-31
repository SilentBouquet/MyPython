import torch
import torch.nn as nn

torch.manual_seed(1)
# batch_first=True：设置输入和输出的张量形状为 [batch_size, sequence_length, input_size]
rnn_layer = nn.RNN(input_size=5, hidden_size=2, num_layers=1, batch_first=True)
W_xh = rnn_layer.weight_ih_l0
W_hh = rnn_layer.weight_hh_l0
b_xh = rnn_layer.bias_ih_l0
b_hh = rnn_layer.bias_hh_l0
print('W_xh shape: ', W_xh.shape)
print('W_hh shape: ', W_hh.shape)
print('b_xh shape: ', b_xh.shape)
print('b_hh shape: ', b_hh.shape)
print()

# 创建一个输入张量，形状为[3, 5]，表示一个长度为3的序列，每个时间步有5个特征
x_seq = torch.tensor([[1.0] * 5, [2.0] * 5, [3.0] * 5]).float()
# output：输出张量，形状为 [1, 3, 2]（每个时间步的输出）。
# hn：最后一个时间步的隐藏状态，形状为 [1, 1, 2]
output, hn = rnn_layer(torch.reshape(x_seq, (1, 3, 5)))
out_man = []
# 计算每个时间步的 RNN 输出
for t in range(3):
    xt = torch.reshape(x_seq[t], (1, 5))
    print(f'Time step {t} =>')
    print('      Input              : ', xt.numpy())
    # 计算输入到隐藏层的线性变换
    ht = torch.matmul(xt, torch.transpose(W_xh, 0, 1)) + b_xh
    # detach() 返回一个与原张量共享数据但不参与求导的张量
    # 使用 detach() 之后，得到的新张量不会被计算图跟踪，因此后续对它的操作不会影响梯度的计算
    print('      Hidden           : ', ht.detach().numpy())
    # 计算隐藏层到隐藏层的部分
    if t > 0:
        prev_h = out_man[t-1]
    else:
        prev_h = torch.zeros(ht.shape)
    ot = ht + torch.matmul(prev_h, torch.transpose(W_hh, 0, 1)) + b_hh
    ot = torch.tanh(ot)
    out_man.append(ot)
    print('      Output           : ', ot.detach().numpy())
    print('      RNN Output   : ', output[:, t].detach().numpy())
    print()