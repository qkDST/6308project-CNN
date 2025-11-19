# 1. 导入
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (Input, Embedding, Conv1D, BatchNormalization,
                                     ReLU, MaxPooling1D, GlobalMaxPooling1D, Dense,
                                     Dropout, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import nltk

# 下载必要资源（仅分词和停用词库）
nltk.download('punkt')
nltk.download('stopwords')

# 2. 配置参数
class Config:
    # 核心：配置Yahoo和Yelp数据集路径
    YAHOO_TRAIN_PATH = "C:/Users/31278/Desktop/text_classification/yahoo_answers_csv/train.csv"
    YAHOO_TEST_PATH = "C:/Users/31278/Desktop/text_classification/yahoo_answers_csv/test.csv"
    YELP_TRAIN_PATH = "C:/Users/31278/Desktop/text_classification/yelp_review_polarity_csv/train.csv"
    YELP_TEST_PATH = "C:/Users/31278/Desktop/text_classification/yelp_review_polarity_csv/test.csv"
    SAVE_MODEL_PATH = "best_models/"  # 最佳模型保存路径
    
    # 数据预处理参数（嵌入层随机初始化）
    SAMPLE_RATIO = 0.2  # 取样20%数据（避免训练过久，可改0.1更快）
    MAX_VOCAB_SIZE = 50000  # 词表最大容量（前5万高频词）
    YAHOO_MAX_LEN = 300  # Yahoo长文本序列长度
    YELP_MAX_LEN = 200  # Yelp短文本序列长度
    EMBEDDING_DIM = 100  # 嵌入层维度（随机初始化）
    
    # 训练参数
    BATCH_SIZE = 64  # 降低批次大小，避免内存不足（根据电脑性能替换64或128）
    EPOCHS = 15  # 减少训练轮次，加快速度
    LEARNING_RATE = 0.001
    PATIENCE = 3  # 早停：3轮无提升则停止

# 创建模型保存文件夹（不存在则自动创建）
os.makedirs(Config.SAVE_MODEL_PATH, exist_ok=True)
config = Config()

# 3. 数据预处理工具函数
def load_yahoo_data(train_path, test_path):
    """加载Yahoo Answers数据集（多分类：10类）"""
    # 读取训练集和测试集（适配新Kaggle数据集的列顺序：category→title→content→answer）
    train_df = pd.read_csv(train_path, header=None, names=['category', 'title', 'content', 'answer'])
    test_df = pd.read_csv(test_path, header=None, names=['category', 'title', 'content', 'answer'])
    # 合并后统一预处理（避免分开处理导致差异）
    df = pd.concat([train_df, test_df], ignore_index=True)
    # 过滤空值（避免预处理报错）
    df = df.dropna(subset=['title', 'content', 'answer', 'category'])
    # 拼接文本（标题+内容+回答，保留完整语义）
    df['text'] = df['title'] + " " + df['content'] + " " + df['answer']
    # 取样20%（分层取样，保持类别分布均匀）
    df_sample, _ = train_test_split(df, test_size=1-config.SAMPLE_RATIO, 
                                    stratify=df['category'], random_state=42)
    # 标签编码（文本类别→整数0-9，适配模型输出）
    label_encoder = LabelEncoder()
    df_sample['label'] = label_encoder.fit_transform(df_sample['category'])
    return df_sample['text'].values, df_sample['label'].values, len(label_encoder.classes_)

def load_yelp_data(train_path, test_path):
    """加载Yelp Review Polarity数据集（二分类：正负评论）"""
    # 读取训练集和测试集（列顺序：label→text）
    train_df = pd.read_csv(train_path, header=None, names=['label', 'text'])
    test_df = pd.read_csv(test_path, header=None, names=['label', 'text'])
    # 合并后统一预处理
    df = pd.concat([train_df, test_df], ignore_index=True)
    # 过滤空值
    df = df.dropna(subset=['text'])
    # 取样20%（分层取样）
    df_sample, _ = train_test_split(df, test_size=1-config.SAMPLE_RATIO, 
                                    stratify=df['label'], random_state=42)
    # 标签转换（1→0负面，2→1正面，适配二分类损失函数）
    df_sample['label'] = df_sample['label'].map({1:0, 2:1})
    return df_sample['text'].values, df_sample['label'].values, 2

def clean_text(text):
    """文本清洗：小写化+去除特殊字符+去冗余空格（小白无需修改）"""
    text = text.lower()  # 统一小写（避免大小写重复计算）
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # 只保留字母和空格（去除标点、数字等）
    text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
    return text

def text_preprocess(texts, max_len, tokenizer=None, fit_tokenizer=True):
    """文本序列化：分词→过滤停用词→序列转换→对齐"""
    stop_words = set(stopwords.words('english'))  # 加载英文停用词（比如the、a等无意义词）
    tokenized_texts = []
    for text in texts:
        cleaned_text = clean_text(text)  # 先清洗文本
        tokens = word_tokenize(cleaned_text)  # 分词（把句子拆成单个单词）
        filtered_tokens = [token for token in tokens if token not in stop_words]  # 过滤停用词
        tokenized_texts.append(filtered_tokens)
    
    # 构建/使用词表（把单词→整数索引）
    if fit_tokenizer:
        tokenizer = Tokenizer(num_words=config.MAX_VOCAB_SIZE, oov_token='<OOV>')  # OOV：未登录词标记
        tokenizer.fit_on_texts(tokenized_texts)  # 基于训练集构建词表
    
    # 文本→整数序列
    sequences = tokenizer.texts_to_sequences(tokenized_texts)
    # 序列对齐（统一长度：长截断、短填充）
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer

# 4. 模型实现（嵌入矩阵，用随机初始化嵌入层）
def build_very_deep_cnn(input_len, num_classes):
    """构建Very Deep CNN模型（无GloVe，嵌入层随机初始化）"""
    # 输入层：(序列长度,)
    inputs = Input(shape=(input_len,), name="input_layer")
    
    # 嵌入层：随机初始化
    embedding = Embedding(
        input_dim=config.MAX_VOCAB_SIZE + 1,  # 词表大小+1（预留索引0）
        output_dim=config.EMBEDDING_DIM,     # 嵌入维度（100维）
        input_length=input_len,              # 输入序列长度
        trainable=True,                      # 训练中可更新（适配任务数据）
        name="embedding_layer"
    )(inputs)
    
    # 卷积块1：3-gram（捕捉短距离语义）
    x = Conv1D(filters=64, kernel_size=3, padding='same', name="conv_3gram")(embedding)
    x = BatchNormalization(name="bn1")(x)  # 批量归一化：加速收敛
    x = ReLU(name="relu1")(x)              # 激活函数：引入非线性
    x = MaxPooling1D(pool_size=2, strides=1, padding='same', name="pool1")(x)  # 池化：保留关键特征
    
    # 卷积块2：5-gram（捕捉中距离语义）
    x = Conv1D(filters=64, kernel_size=5, padding='same', name="conv_5gram")(x)
    x = BatchNormalization(name="bn2")(x)
    x = ReLU(name="relu2")(x)
    x = MaxPooling1D(pool_size=2, strides=1, padding='same', name="pool2")(x)
    
    # 卷积块3：3-gram（增强短距离特征）
    x = Conv1D(filters=64, kernel_size=3, padding='same', name="conv_3gram_2")(x)
    x = BatchNormalization(name="bn3")(x)
    x = ReLU(name="relu3")(x)
    x = MaxPooling1D(pool_size=2, strides=1, padding='same', name="pool3")(x)
    
    # 卷积块4：7-gram（捕捉长距离语义，适配Yahoo长文本）
    x = Conv1D(filters=64, kernel_size=7, padding='same', name="conv_7gram")(x)
    x = BatchNormalization(name="bn4")(x)
    x = ReLU(name="relu4")(x)
    
    # 全局最大池化：将变长序列→固定长度向量
    x = GlobalMaxPooling1D(name="global_pool")(x)
    x = Dropout(0.5, name="dropout")(x)  # Dropout：缓解过拟合
    
    # 输出层：多分类用softmax，二分类用sigmoid
    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    outputs = Dense(num_classes, activation=activation, name="output_layer")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Very_Deep_CNN")
    return model

def dense_block(x, num_layers, growth_rate, block_name):
    """DenseNet密集块：特征复用（小白无需理解，直接使用）"""
    features = [x]
    for i in range(num_layers):
        # 瓶颈层：1x1卷积降维，减少计算量
        bottleneck = Conv1D(
            filters=4 * growth_rate,
            kernel_size=1,
            padding='same',
            name=f"{block_name}_bottleneck_{i}"
        )(Concatenate(name=f"{block_name}_concat_{i}")(features))
        bottleneck = BatchNormalization(name=f"{block_name}_bn_bottleneck_{i}")(bottleneck)
        bottleneck = ReLU(name=f"{block_name}_relu_bottleneck_{i}")(bottleneck)
        
        # 特征提取层：3x1卷积
        conv = Conv1D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            name=f"{block_name}_conv_{i}"
        )(bottleneck)
        conv = BatchNormalization(name=f"{block_name}_bn_conv_{i}")(conv)
        conv = ReLU(name=f"{block_name}_relu_conv_{i}")(conv)
        
        features.append(conv)  # 新增特征加入复用列表
    
    return Concatenate(name=f"{block_name}_final_concat")(features)

def transition_layer(x, compression, layer_name):
    """DenseNet过渡层：压缩特征维度"""
    num_features = x.shape[-1]
    x = Conv1D(
        filters=int(num_features * compression),
        kernel_size=1,
        padding='same',
        name=f"{layer_name}_conv"
    )(x)
    x = BatchNormalization(name=f"{layer_name}_bn")(x)
    x = ReLU(name=f"{layer_name}_relu")(x)
    x = MaxPooling1D(pool_size=2, padding='same', name=f"{layer_name}_pool")(x)
    return x

def build_text_densenet(input_len, num_classes):
    """构建Text DenseNet模型（嵌入层随机初始化）"""
    # 输入层
    inputs = Input(shape=(input_len,), name="input_layer")
    
    # 嵌入层：随机初始化
    embedding = Embedding(
        input_dim=config.MAX_VOCAB_SIZE + 1,
        output_dim=config.EMBEDDING_DIM,
        input_length=input_len,
        trainable=True,
        name="embedding_layer"
    )(inputs)
    
    # 初始卷积层：将嵌入向量→特征图
    x = Conv1D(filters=32, kernel_size=3, padding='same', name="init_conv")(embedding)
    x = BatchNormalization(name="init_bn")(x)
    x = ReLU(name="init_relu")(x)
    
    # 密集块1 + 过渡层1
    x = dense_block(x, num_layers=3, growth_rate=16, block_name="dense_block1")
    x = transition_layer(x, compression=0.5, layer_name="transition1")
    
    # 密集块2 + 过渡层2
    x = dense_block(x, num_layers=3, growth_rate=16, block_name="dense_block2")
    x = transition_layer(x, compression=0.5, layer_name="transition2")
    
    # 全局池化 + Dropout
    x = GlobalMaxPooling1D(name="global_pool")(x)
    x = Dropout(0.5, name="dropout")(x)
    
    # 输出层
    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    outputs = Dense(num_classes, activation=activation, name="output_layer")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Text_DenseNet")
    return model

# 5. 训练与评估工具函数
def train_model(model, X_train, y_train, X_val, y_val, num_classes, model_name):
    """训练模型：含早停、学习率调度、保存最佳模型"""
    # 选择损失函数：多分类→稀疏交叉熵，二分类→二元交叉熵
    loss_fn = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    
    # 编译模型（优化器用Adam，adamw另论）
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy']  # 训练时监控准确率
    )
    
    # 早停：避免过拟合（3轮验证损失无提升则停止）
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # 学习率调度：验证损失停滞时，学习率减半
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,  # 最小学习率（避免过小导致不收敛）
        verbose=1
    )
    
    # 保存最佳模型（按验证准确率）
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(config.SAVE_MODEL_PATH, f"{model_name}_best.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # 开始训练
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, lr_scheduler, model_checkpoint],
        shuffle=True  # 训练集打乱，提升泛化能力
    )
    return history

def evaluate_model(model, X_test, y_test, num_classes, model_name, dataset_name):
    """评估模型：计算核心指标+可视化混淆矩阵"""
    # 预测结果
    y_pred_proba = model.predict(X_test, verbose=0)  # 预测概率
    if num_classes > 2:
        y_pred = np.argmax(y_pred_proba, axis=1)  # 多分类：取概率最大的类别
        # 多分类指标
        accuracy = accuracy_score(y_test, y_pred)
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='micro', zero_division=0
        )
        # 打印结果（可直接看准确率和F1）
        print(f"\n【{dataset_name} - {model_name} 多分类评估结果】")
        print(f"准确率：{accuracy:.4f}")
        print(f"宏平均F1：{macro_f1:.4f}")
        print(f"微平均F1：{micro_f1:.4f}")
    else:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()  # 二分类：阈值0.5
        # 二分类指标
        accuracy = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        auc = roc_auc_score(y_test, y_pred_proba)
        # 打印结果
        print(f"\n【{dataset_name} - {model_name} 二分类评估结果】")
        print(f"准确率：{accuracy:.4f}")
        print(f"F1分数：{f1:.4f}")
        print(f"AUC：{auc:.4f}")
    
    # 可视化混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f"类{i}" for i in range(num_classes)],
                yticklabels=[f"类{i}" for i in range(num_classes)])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f"{dataset_name} - {model_name} 混淆矩阵")
    plt.savefig(f"{model_name}_{dataset_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy

def plot_training_history(history, model_name, dataset_name):
    """可视化训练历史：损失+准确率曲线（判断模型是否收敛）"""
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('轮次（Epoch）')
    plt.ylabel('损失（Loss）')
    plt.title(f"{dataset_name} - {model_name} 损失曲线")
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.xlabel('轮次（Epoch）')
    plt.ylabel('准确率（Accuracy）')
    plt.title(f"{dataset_name} - {model_name} 准确率曲线")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_{dataset_name}_training_history.png", dpi=300, bbox_inches='tight')
    plt.show()

# 6. 主函数（执行全流程：加载数据→预处理→训练→评估）
def main():
    # 步骤1：加载Yahoo数据集
    print("="*50)
    print("1. 加载Yahoo Answers数据集...")
    yahoo_texts, yahoo_labels, yahoo_num_classes = load_yahoo_data(
        config.YAHOO_TRAIN_PATH, config.YAHOO_TEST_PATH
    )
    # 文本序列化（训练集构建词表）
    yahoo_sequences, yahoo_tokenizer = text_preprocess(
        yahoo_texts, max_len=config.YAHOO_MAX_LEN, fit_tokenizer=True
    )
    # 划分训练集/验证集/测试集（8:1:1）
    X_train_yahoo, X_temp_yahoo, y_train_yahoo, y_temp_yahoo = train_test_split(
        yahoo_sequences, yahoo_labels, test_size=0.2, stratify=yahoo_labels, random_state=42
    )
    X_val_yahoo, X_test_yahoo, y_val_yahoo, y_test_yahoo = train_test_split(
        X_temp_yahoo, y_temp_yahoo, test_size=0.5, stratify=y_temp_yahoo, random_state=42
    )
    print(f"Yahoo数据集准备完成：训练集{len(X_train_yahoo)}条，验证集{len(X_val_yahoo)}条，测试集{len(X_test_yahoo)}条")

    # 步骤2：加载Yelp数据集
    print("\n" + "="*50)
    print("2. 加载Yelp Review Polarity数据集...")
    yelp_texts, yelp_labels, yelp_num_classes = load_yelp_data(
        config.YELP_TRAIN_PATH, config.YELP_TEST_PATH
    )
    # 文本序列化
    yelp_sequences, yelp_tokenizer = text_preprocess(
        yelp_texts, max_len=config.YELP_MAX_LEN, fit_tokenizer=True
    )
    # 划分训练集/验证集/测试集（8:1:1）
    X_train_yelp, X_temp_yelp, y_train_yelp, y_temp_yelp = train_test_split(
        yelp_sequences, yelp_labels, test_size=0.2, stratify=yelp_labels, random_state=42
    )
    X_val_yelp, X_test_yelp, y_val_yelp, y_test_yelp = train_test_split(
        X_temp_yelp, y_temp_yelp, test_size=0.5, stratify=y_temp_yelp, random_state=42
    )
    print(f"Yelp数据集准备完成：训练集{len(X_train_yelp)}条，验证集{len(X_val_yelp)}条，测试集{len(X_test_yelp)}条")

    # 步骤3：训练Very Deep CNN模型
    print("\n" + "="*50)
    print("3. 训练Very Deep CNN模型（Yahoo数据集）...")
    vdcnn_yahoo = build_very_deep_cnn(
        input_len=config.YAHOO_MAX_LEN,
        num_classes=yahoo_num_classes
    )
    vdcnn_yahoo_history = train_model(
        vdcnn_yahoo, X_train_yahoo, y_train_yahoo, X_val_yahoo, y_val_yahoo,
        num_classes=yahoo_num_classes, model_name="Very_Deep_CNN_Yahoo"
    )
    # 评估模型
    evaluate_model(vdcnn_yahoo, X_test_yahoo, y_test_yahoo, yahoo_num_classes, "Very_Deep_CNN", "Yahoo")
    # 可视化训练历史
    plot_training_history(vdcnn_yahoo_history, "Very_Deep_CNN", "Yahoo")

    print("\n" + "="*50)
    print("4. 训练Very Deep CNN模型（Yelp数据集）...")
    vdcnn_yelp = build_very_deep_cnn(
        input_len=config.YELP_MAX_LEN,
        num_classes=yelp_num_classes
    )
    vdcnn_yelp_history = train_model(
        vdcnn_yelp, X_train_yelp, y_train_yelp, X_val_yelp, y_val_yelp,
        num_classes=yelp_num_classes, model_name="Very_Deep_CNN_Yelp"
    )
    evaluate_model(vdcnn_yelp, X_test_yelp, y_test_yelp, yelp_num_classes, "Very_Deep_CNN", "Yelp")
    plot_training_history(vdcnn_yelp_history, "Very_Deep_CNN", "Yelp")

    # 步骤4：训练Text DenseNet模型
    print("\n" + "="*50)
    print("5. 训练Text DenseNet模型（Yahoo数据集）...")
    densenet_yahoo = build_text_densenet(
        input_len=config.YAHOO_MAX_LEN,
        num_classes=yahoo_num_classes
    )
    densenet_yahoo_history = train_model(
        densenet_yahoo, X_train_yahoo, y_train_yahoo, X_val_yahoo, y_val_yahoo,
        num_classes=yahoo_num_classes, model_name="Text_DenseNet_Yahoo"
    )
    evaluate_model(densenet_yahoo, X_test_yahoo, y_test_yahoo, yahoo_num_classes, "Text_DenseNet", "Yahoo")
    plot_training_history(densenet_yahoo_history, "Text_DenseNet", "Yahoo")

    print("\n" + "="*50)
    print("6. 训练Text DenseNet模型（Yelp数据集）...")
    densenet_yelp = build_text_densenet(
        input_len=config.YELP_MAX_LEN,
        num_classes=yelp_num_classes
    )
    densenet_yelp_history = train_model(
        densenet_yelp, X_train_yelp, y_train_yelp, X_val_yelp, y_val_yelp,
        num_classes=yelp_num_classes, model_name="Text_DenseNet_Yelp"
    )
    evaluate_model(densenet_yelp, X_test_yelp, y_test_yelp, yelp_num_classes, "Text_DenseNet", "Yelp")
    plot_training_history(densenet_yelp_history, "Text_DenseNet", "Yelp")

    print("\n" + "="*50)
    print("🎉 所有模型训练与评估完成！")
    print(f"模型权重保存在：{config.SAVE_MODEL_PATH}")
    print(f"可视化图表保存在：当前项目文件夹（.png文件）")

# 执行主函数
if __name__ == "__main__":

    main()
