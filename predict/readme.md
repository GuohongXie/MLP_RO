# 代码目录结构
```
predict/
|   readme.md
|
+---old_predict/
|
+---predict_all/
|       all_oneNet_mlp_4_layer_struct_0x1.py
|
+---predict_functions/
|   |   function_predict.py
|   |   predict_all_data_with_one_net.py
|   |   predict_all_data_with_three_net.py
|   |   predict_layer_data_with_one_net.py
|   |   predict_layer_data_with_three_net.py
|   |   __init__.py
|   |
|   \---__pycache__/
|
+---predict_inside/
|       inside_oneNet_mlp_4_layer_struct_0x1.py
|       inside_threeNet_mlp_4_layer_struct_0x1.py
|       inside_threeNet_mlp_4_layer_struct_0x2.py
|       inside_threeNet_mlp_4_layer_struct_0x3.py
|       inside_threeNet_mlp_4_layer_struct_0x4.py
|       inside_threeNet_mlp_4_layer_struct_0x5.py
|
\---predict_outside/
        outside_oneNet_mlp_4_layer_struct_0x1.py
        outside_threeNet_mlp_4_layer_struct_0x1.py
        outside_threeNet_mlp_4_layer_struct_0x2.py
        outside_threeNet_mlp_4_layer_struct_0x3.py
        outside_threeNet_mlp_4_layer_struct_0x4.py
```


# 代码目录说明
./old_predict/ 为 CAE 会议版本的预测代码，已废弃，留作备份
./predict_functions/ 为预测 U/P/C 的函数
./predict_all/ 为不分层训练时的预测代码
./predict_inside/ 为主体层的预测代码
././predict_outside/ 为浓差极化层的预测代码




# 注意事项
1.请勿改动代码目录结构和名称，否则代码也需要做出相应的改变


