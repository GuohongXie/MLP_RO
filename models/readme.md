# 代码目录结构
```
    models/
        |   mlp_128_64_32_1.py
        |   mlp_128_64_32_1_LastRelu.py
        |   mlp_128_64_32_3.py
        |   mlp_resnet_64_64_64_64_64_1.py
        |   mlp_resnet_64_64_64_64_64_1_LastRelu.py
        |   readme.md
        |   __init__.py
        |
        \---__pycache__/
    
```


# 代码目录说明
mlp 即为 MLP 模型
后面的数字表示各隐藏层和输出层的神经元数
LastRelu表示在输出层再加一层Relu激活函数
mlp_resnet 表示添加了残差链接的 MLP 模型，具体结构见代码

如果有需要也可以自己编写添加新的神经网络模型代码，注意添加完模型后需要修改 "__init__.py" 文件


# 注意事项
1.请勿改动代码目录结构和名称，否则代码也需要做出相应的改变


