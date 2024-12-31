import torch

logits = torch.tensor([0.8])
probas = torch.sigmoid(logits)
target = torch.tensor([1.0])
BCE_loss_func = torch.nn.BCELoss()
BCE_logits_loss_func = torch.nn.BCEWithLogitsLoss()
print(f'BCE (w Probas): {BCE_loss_func(probas, target):.4f}')
print(f'BCE (w Logits): {BCE_logits_loss_func(logits, target):.4f}')

logits = torch.tensor([[1.5, 0.8, 2.1]])
probas = torch.softmax(logits, dim=1)
target = torch.tensor([2])
CCE_loss_func = torch.nn.NLLLoss()
CCE_logits_loss_func = torch.nn.CrossEntropyLoss()
print(f'CCE (w Probas): {CCE_loss_func(torch.log(probas), target):.4f}')
print(f'CCE (w Logits): {CCE_logits_loss_func(logits, target):.4f}')