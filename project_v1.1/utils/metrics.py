import torch


def calculate_accuracy(outputs, targets):
    """计算准确率"""
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total


def calculate_precision_recall_f1(outputs, targets, num_classes):
    """计算精确率、召回率和F1分数"""
    _, predicted = torch.max(outputs, 1)

    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1 = torch.zeros(num_classes)

    for i in range(num_classes):
        true_positive = ((predicted == i) & (targets == i)).sum().float()
        false_positive = ((predicted == i) & (targets != i)).sum().float()
        false_negative = ((predicted != i) & (targets == i)).sum().float()

        precision[i] = true_positive / (true_positive + false_positive + 1e-8)
        recall[i] = true_positive / (true_positive + false_negative + 1e-8)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)

    return precision.mean().item(), recall.mean().item(), f1.mean().item()