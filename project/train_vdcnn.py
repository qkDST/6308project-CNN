import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os

from models.vdcnn import VDCNN
from data.preprocess import TextDataset
from utils.config import Config
from utils.metrics import calculate_accuracy


def train_vdcnn(dataset_type='yahoo'):
    """训练VDCNN模型"""
    config = Config()

    # 设置数据集参数
    if dataset_type == 'yahoo':
        num_classes = config.num_classes_yahoo
        train_file = r'E:\study\hkmaster\6308\yahoo_answers_csv\train.csv'  # 请替换为实际路径
        test_file = r'E:\study\hkmaster\\6308\yahoo_answers_csv\test.csv'  # 请替换为实际路径
    else:  # yelp
        num_classes = config.num_classes_yelp
        train_file = r'E:\study\hkmaster\6308\yelp_review_polarity_csv\test.csv'  # 请替换为实际路径
        test_file = r'E:\study\hkmaster\6308\yelp_review_polarity_csv\train.csv'  # 请替换为实际路径

    # 数据加载器
    train_dataset = TextDataset(train_file, dataset_type, config.max_length, is_train=True)
    test_dataset = TextDataset(test_file, dataset_type, config.max_length, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers)

    # 模型
    model = VDCNN(num_classes=num_classes, depth=config.vdcnn_depth,
                  use_shortcut=config.vdcnn_use_shortcut)
    model.to(config.device)

    # 优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                          momentum=config.momentum, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_accuracy = 0.0
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

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_acc = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(config.device), target.to(config.device)
                output = model(data)
                test_loss += criterion(output, target).item()
                test_acc += calculate_accuracy(output, target)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

        print(f'\nEpoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), f'vdcnn_{dataset_type}_best.pth')

    print(f'Best Test Accuracy: {best_accuracy:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['yahoo', 'yelp'],
                        default='yahoo', help='Dataset to use')
    args = parser.parse_args()

    train_vdcnn(args.dataset)