import torch


#定义预测函数
def function_predict(row, model_used_2, device_2):
    # convert row to data
    row = torch.Tensor([row])
    #row = row.to(device_2)
    # make prediction
    y_hat = model_used_2(row)
    #y_hat = y_hat.cpu()
    # retrieve numpy array
    y_hat = y_hat.detach().numpy()
    return y_hat

