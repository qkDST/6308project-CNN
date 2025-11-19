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
