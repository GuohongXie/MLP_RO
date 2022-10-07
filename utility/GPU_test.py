import torch
print(torch.cuda.is_available())       # 判断 GPU 是否可用
print(torch.cuda.device_count())       # 判断有多少 GPU
print(torch.cuda.get_device_name(0))          # 返回 gpu 名字，设备索引默认从 0 开始
print(torch.cuda.current_device())     # 返回当前设备索引