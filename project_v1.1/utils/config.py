import torch
import os


class Config:
    #dataset path
    data_base_path = r'E:\study\hkmaster\6308'

    # Yahoo
    yahoo_train_path = os.path.join(data_base_path, 'yahoo_answers_csv/train.csv')
    yahoo_test_path = os.path.join(data_base_path, 'yahoo_answers_csv/test.csv')

    # Yelp
    yelp_train_path = os.path.join(data_base_path, 'yelp_review_polarity_csv/train.csv')
    yelp_test_path = os.path.join(data_base_path, 'yelp_review_polarity_csv/test.csv')


    max_length = 512
    batch_size = 16
    num_workers = 0

    data_fraction = 0.1

    #characters setting
    vocab_size = 69
    char_embedding_dim = 16

    # model setting
    num_classes_yahoo = 10
    num_classes_yelp = 2
    dropout_rate = 0.2

    #training setting
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 50  #numbers of training circulation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Early Stopping
    early_stopping_patience = 5  #
    early_stopping_delta = 0.001  # minimum change
    early_stopping_criterion = 'accuracy'  # 'loss' 或 'accuracy'

    # VDCNN
    vdcnn_depth = 9
    vdcnn_use_shortcut = True

    # DenseNet
    densenet_growth_rate = 12
    densenet_block_config = (3, 6, 12)
    densenet_compression = 0.5
    densenet_bottleneck = True