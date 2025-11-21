import torch


class Config:
    # 数据配置
    max_length = 512  # VDCNN论文中使用的固定长度
    batch_size = 16
    num_workers = 0

    # 字符级配置（VDCNN使用）
    vocab_size = 69  # VDCNN论文中的69个字符
    char_embedding_dim = 16

    # 模型通用配置
    num_classes_yahoo = 10
    num_classes_yelp = 2
    dropout_rate = 0.2

    # 训练配置
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 5  #训练轮数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # VDCNN特定配置
    vdcnn_depth = 9  # 9, 17, 29, 49
    vdcnn_use_shortcut = True

    # DenseNet文本特定配置
    densenet_growth_rate = 12
    densenet_block_config = (3,6,12)
    densenet_compression = 0.5
    densenet_bottleneck = True