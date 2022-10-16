import torch


#定义预测函数
def function_predict(row, model_used_2):
    row = torch.Tensor([row])
    y_hat = model_used_2(row)
    y_hat = y_hat.detach().numpy()
    return y_hat

