import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
import time

from models.densenet_text import DenseNetText
from data.preprocess import TextDataset
from utils.config import Config
from utils.metrics import calculate_accuracy
from utils.early_stopping import EarlyStopping


def train_densenet(dataset_type='yahoo'):
    config = Config()

    #dataset parameter
    if dataset_type == 'yahoo':
        num_classes = config.num_classes_yahoo
        train_file = config.yahoo_train_path
        test_file = config.yahoo_test_path
    else:  # yelp
        num_classes = config.num_classes_yelp
        train_file = config.yelp_train_path
        test_file = config.yelp_test_path

    print(f"Start training {dataset_type} dataset")
    #print(f"Early Stopping patience: {config.early_stopping_patience}")

    # 加载训练数据 - 两个数据集都使用前10%
    full_train_dataset = TextDataset(train_file, dataset_type, config.max_length,
                                     is_train=True, data_fraction=config.data_fraction)

    # 在取出的数据上划分训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

    # 测试集也使用相同比例的数据
    test_dataset = TextDataset(test_file, dataset_type, config.max_length,
                               is_train=False, data_fraction=config.data_fraction)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers)

    print(f"Datasets used: {len(full_train_dataset)}")
    print(f"Training set: {len(train_subset)}")
    print(f"Validation set: {len(val_subset)}")
    print(f"Testing set: {len(test_dataset)}")

    # model
    model = DenseNetText(
        num_classes=num_classes,
        growth_rate=config.densenet_growth_rate,
        block_config=config.densenet_block_config,
        compression=config.densenet_compression,
        dropout_rate=config.dropout_rate
    )
    model.to(config.device)

    #print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                          momentum=config.momentum, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Early Stopping
    checkpoint_path = f'densenet_{dataset_type}_best.pth'
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        verbose=True,
        delta=config.early_stopping_delta,
        path=checkpoint_path
    )

    # 训练循环
    start_time = time.time()

    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(output, target)

            #training prcess
            """if batch_idx % 50 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')"""

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config.device), target.to(config.device)
                output = model(data)
                val_loss += criterion(output, target).item()
                val_acc += calculate_accuracy(output, target)

        # 计算平均损失和准确率
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f'\nEpoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')

        # Early Stopping检查
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print(f'\nTraining Compeleted, Time: {total_time:.2f}s')
    print(f'Trained  {epoch + 1}  epoches')

    # 加载最佳模型进行测试
    print(f"Loading best model: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))

    # 在测试集上评估
    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            test_loss += criterion(output, target).item()
            test_acc += calculate_accuracy(output, target)

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    print(f'\nTesting result:')
    print(f'Loss: {test_loss:.4f}, Acc: {test_acc:.4f}')

    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['yahoo', 'yelp'],
                        default='yahoo', help='Dataset to use')
    args = parser.parse_args()

    train_densenet(args.dataset)