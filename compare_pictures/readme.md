
# 代码目录结构
```
compare_pictures/
        |   pic_comparing.py
        |   readme.md
        |
        +---pictures_compare_functions/
        |   |   mse_grey.py
        |   |   psnr.py
        |   |   ssim_grey.py
        |   |   ssim_multi_channel.py
        |   |   __init__.py
        |   |
        |   \---__pycache__/
        |
        \---results/
                inside_plus_outside_oneNet_mlp_4_layer_struct_0x1.csv
                inside_plus_outside_threeNet_mlp_4_layer_struct_0x1.csv
                inside_plus_outside_threeNet_mlp_4_layer_struct_0x2.csv
                inside_plus_outside_threeNet_mlp_4_layer_struct_0x3.csv
                inside_plus_outside_threeNet_mlp_4_layer_struct_0x4.csv
                out_p_test_0x3.csv

```


# 代码目录说明
./pic_comparing.py 为主函数代码，运行时需要运行此代码文件
./pictures_compare_functions/ 为各种图片相似度对比函数
./pictures_compare_functions/__pycache__/文件夹可以忽略
./results/ 为 CFD 图像与多种 MLP 预测图像对比结果



# 注意事项
1.请勿改动代码目录结构和名称，否则代码也需要做出相应的改变


