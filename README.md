# 6308project-CNN
# 配置环境，VScode下载Python  Pylance   Code Runner三个插件，cmd安装pytho依赖库  pip install tensorflow==2.10 pandas numpy nltk scikit-learn matplotlib seaborn
# 修改文件路径 
# yelp文件的数据库太大了没法上传
基于 Yahoo Answers（多分类）和 Yelp Review Polarity（二分类）数据集，使用 Very Deep CNN 和 Text DenseNet 两种模型实现文本分类任务。
Yahoo Answers：多分类任务（10个类别），下载地址：https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset
Yelp Review Polarity：二分类任务（正负评论），下载地址：https://www.kaggle.com/irustandi/yelp-review-polarity/version/1
模型架构：
1. Very Deep CNN：4个卷积块（3-gram/5-gram/3-gram/7-gram）+ 全局最大池化 + Dropout
2. Text DenseNet：2个密集块 + 过渡层 + 全局最大池化 + Dropout


#11.24更新by黄梓昊
pytorch版本更新版存放于project_v1.1中
添加了earlystop以便及时停止训练
并去了一丢丢代码的ai味
为了加快运行速度规定了每个数据集只读前10%（几十万几百万的数据真的会跑一天）

现存问题：
准确度不太高，可能还需要调整模型参数


# 11.22更新by黄梓昊
新增借助deepseek之力搓了个pytorch环境下的版本
在project文件夹内

与tensorflow类似，pytorch库安装教程：https://www.runoob.com/pytorch/pytorch-install.html
# 文件夹内部结构：
models/: 模型定义
data/: 数据加载和预处理
utils/: 配置和工具函数
训练脚本和依赖文件
# 使用前请注意：
请将数据集文件路径替换为实际路径

确保有足够的GPU内存用于训练深度模型

可根据需要调整超参数

# 训练指令：在命令行中输入以下命令
# 训练Yahoo数据集
python train_vdcnn.py --dataset yahoo

# 训练Yelp数据集  
python train_vdcnn.py --dataset yelp

# 训练Yahoo数据集
python train_densenet.py --dataset yahoo

# 训练Yelp数据集
python train_densenet.py --dataset yelp
