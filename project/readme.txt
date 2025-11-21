因为tensorflow环境配置比较复杂，自己电脑配出bug了所以借助deepseek之力搓了个pytorch环境下的版本

pytorch安装教程：https://www.runoob.com/pytorch/pytorch-install.html

models/: 模型定义

data/: 数据加载和预处理

utils/: 配置和工具函数

训练脚本和依赖文件

请将数据集文件路径替换为实际路径

确保有足够的GPU内存用于训练深度模型

可根据需要调整超参数
训练指令：在命令行中输入以下命令
# 训练Yahoo数据集
python train_vdcnn.py --dataset yahoo

# 训练Yelp数据集  
python train_vdcnn.py --dataset yelp

# 训练Yahoo数据集
python train_densenet.py --dataset yahoo

# 训练Yelp数据集
python train_densenet.py --dataset yelp