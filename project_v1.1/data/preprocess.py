import pandas as pd
import torch
from torch.utils.data import Dataset
import re


class CharTokenizer:


    def __init__(self):
        # VDCNN论文中的69个字符
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
        self.pad_char = '{'  # 使用不在字符集中的字符作为padding
        self.unk_char = '}'  # 未知字符

        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.char_to_idx[self.pad_char] = len(self.chars)
        self.char_to_idx[self.unk_char] = len(self.chars) + 1

        self.vocab_size = len(self.char_to_idx)

    def encode(self, text, max_length=512):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower().strip()#全小写

        #保留有效字符
        text = ''.join([c for c in text if c in self.char_to_idx or c == ' '])

        encoded = []
        for char in text[:max_length]:
            encoded.append(self.char_to_idx.get(char, self.char_to_idx[self.unk_char]))

        # 填充到固定长度
        if len(encoded) < max_length:
            encoded.extend([self.char_to_idx[self.pad_char]] * (max_length - len(encoded)))

        return encoded


class TextDataset(Dataset):
    def __init__(self, csv_file, dataset_type='yahoo', max_length=512, is_train=True, data_fraction=1.0):
        self.dataset_type = dataset_type
        self.max_length = max_length

        try:
            # 无表头读取CSV
            full_data = pd.read_csv(csv_file, header=None)

            # 只对Yahoo数据集使用前10%，Yelp使用全部数据
            if data_fraction < 1.0:
                sample_size = int(len(full_data) * data_fraction)
                # 按顺序取前10%的数据
                self.data = full_data.head(sample_size).reset_index(drop=True)
                print(
                    f"Reading top {data_fraction * 100}% : {len(self.data)} samples (Totally {len(full_data)} samples)")
            else:
                self.data = full_data
                if dataset_type == 'yahoo':
                    print(f"Yahoo: All data: {len(self.data)} samples")
                else:
                    print(f"Yelp: All data: {len(self.data)} samples")

        except Exception as e:
            print(f"Cannot load {csv_file}: {e}")
            raise

        self.tokenizer = CharTokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        try:
            if self.dataset_type == 'yahoo':
                # Yahoo数据集：4列 - class_index, question_title, question_content, best_answer
                if len(row) >= 4:
                    text = f"{row[1]} {row[2]} {row[3]}"  # 组合标题、内容和答案
                    label = int(row[0]) - 1  # 转换为0-based
                else:
                    # 如果列数不够，使用第一列作为标签，其他列合并为文本
                    text = " ".join([str(x) for x in row[1:]])
                    label = int(row[0]) - 1

            else:  # yelp
                # Yelp数据集：2列 - class_index, review_text
                if len(row) >= 2:
                    text = str(row[1])  # 第二列为评论文本
                    label = int(row[0]) - 1  # 转换为0-based
                else:
                    # 如果只有一列，假设它是文本，标签设为0
                    text = str(row[0])
                    label = 0

        except Exception as e:
            print(f" Error in {idx} : {e}")
            print(f"data: {row}")
            # 返回默认值
            text = "default text"
            label = 0

        # 编码文本
        encoded = self.tokenizer.encode(text, self.max_length)

        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)