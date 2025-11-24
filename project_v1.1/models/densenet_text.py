import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """DenseNet层"""

    def __init__(self, num_input_features, growth_rate, bn_size, dropout_rate):
        super(DenseLayer, self).__init__()

        # 瓶颈层（1x1卷积）
        self.bn1 = nn.BatchNorm1d(num_input_features)
        self.conv1 = nn.Conv1d(num_input_features, bn_size * growth_rate, 1, bias=False)

        # 主卷积层（3x3卷积）
        self.bn2 = nn.BatchNorm1d(bn_size * growth_rate)
        self.conv2 = nn.Conv1d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 瓶颈层
        out = self.conv1(F.relu(self.bn1(x)))
        # 主卷积层
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.dropout(out)

        # 与输入拼接
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(nn.Module):
    """DenseNet块"""

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, dropout_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                dropout_rate
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """过渡层"""

    def __init__(self, num_input_features, num_output_features):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm1d(num_input_features)
        self.conv = nn.Conv1d(num_input_features, num_output_features, 1, bias=False)
        self.pool = nn.AvgPool1d(2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x


class DenseNetText(nn.Module):
    """DenseNet for Text Classification"""

    def __init__(self, num_classes=10, growth_rate=12, block_config=(6, 12, 24),
                 compression=0.5, dropout_rate=0.2, bn_size=4):
        super(DenseNetText, self).__init__()

        # 字符嵌入层
        self.embedding = nn.Embedding(71, 16)

        # 初始卷积
        num_features = 2 * growth_rate
        self.conv1 = nn.Conv1d(16, num_features, 7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)

        # Dense blocks
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            # Dense block
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_rate=dropout_rate
            )
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate

            # Transition layer（除了最后一个block）
            if i != len(block_config) - 1:
                trans = TransitionLayer(
                    num_input_features=num_features,
                    num_output_features=int(num_features * compression)
                )
                self.transition_layers.append(trans)
                num_features = int(num_features * compression)

        # 最终batch norm
        self.final_bn = nn.BatchNorm1d(num_features)

        # 分类器
        self.classifier = nn.Linear(num_features, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 字符嵌入
        x = self.embedding(x)  # [batch, seq_len, 16]
        x = x.transpose(1, 2)  # [batch, 16, seq_len]

        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Dense blocks + transition layers
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        # 全局平均池化
        x = F.relu(self.final_bn(x))
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)

        # 分类
        x = self.classifier(x)

        return x