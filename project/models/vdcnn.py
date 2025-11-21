import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """VDCNN卷积块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, shortcut=False):
        super(ConvBlock, self).__init__()
        self.shortcut = shortcut and (in_channels == out_channels)

        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if shortcut and in_channels != out_channels:
            self.shortcut_conv = nn.Conv1d(in_channels, out_channels, 1)
            self.shortcut_bn = nn.BatchNorm1d(out_channels)
        else:
            self.shortcut_conv = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shortcut:
            if self.shortcut_conv is not None:
                identity = self.shortcut_bn(self.shortcut_conv(identity))
            out += identity

        return F.relu(out)


class VDCNN(nn.Module):
    """Very Deep CNN for Text Classification"""

    def __init__(self, num_classes=10, depth=29, use_shortcut=True):
        super(VDCNN, self).__init__()

        # 字符嵌入层
        self.embedding = nn.Embedding(71, 16)  # 69字符 + padding + unknown

        # 初始卷积层
        self.conv1 = nn.Conv1d(16, 64, 3, padding=1)

        # 卷积块配置（基于深度）
        if depth == 9:
            block_config = [2, 2, 2, 2]  # 2+2+2+2+1=9
        elif depth == 17:
            block_config = [4, 4, 4, 4]  # 4+4+4+4+1=17
        elif depth == 29:
            block_config = [10, 10, 4, 4]  # 10+10+4+4+1=29
        elif depth == 49:
            block_config = [16, 16, 10, 6]  # 16+16+10+6+1=49
        else:
            raise ValueError("Depth must be 9, 17, 29, or 49")

        # 卷积块
        self.blocks = nn.ModuleList()
        current_channels = 64

        # 第一组块 (64通道)
        for i in range(block_config[0]):
            self.blocks.append(ConvBlock(current_channels, 64, shortcut=use_shortcut))

        # 下采样 + 第二组块 (128通道)
        self.blocks.append(nn.Conv1d(64, 128, 3, stride=2, padding=1))
        current_channels = 128
        for i in range(block_config[1]):
            self.blocks.append(ConvBlock(current_channels, 128, shortcut=use_shortcut))

        # 下采样 + 第三组块 (256通道)
        self.blocks.append(nn.Conv1d(128, 256, 3, stride=2, padding=1))
        current_channels = 256
        for i in range(block_config[2]):
            self.blocks.append(ConvBlock(current_channels, 256, shortcut=use_shortcut))

        # 下采样 + 第四组块 (512通道)
        self.blocks.append(nn.Conv1d(256, 512, 3, stride=2, padding=1))
        current_channels = 512
        for i in range(block_config[3]):
            self.blocks.append(ConvBlock(current_channels, 512, shortcut=use_shortcut))

        # K-max池化 (k=8)
        self.kmax_pool = nn.AdaptiveMaxPool1d(8)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        # 字符嵌入 [batch, seq_len] -> [batch, seq_len, 16]
        x = self.embedding(x)

        # 转置为 [batch, 16, seq_len] 用于1D卷积
        x = x.transpose(1, 2)

        # 初始卷积
        x = F.relu(self.conv1(x))

        # 通过所有块
        for block in self.blocks:
            x = block(x)

        # K-max池化
        x = self.kmax_pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 分类
        x = self.classifier(x)

        return x