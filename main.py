# 导入必要库
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (Input, Embedding, Conv1D, BatchNormalization,
                                     ReLU, MaxPooling1D, GlobalMaxPooling1D, Dense,
                                     Dropout, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import nltk

# 配置字体（解决乱码问题）
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams['figure.figsize'] = (10, 6)  # 默认图形大小

# 下载NLTK资源
nltk.download('punkt')
nltk.download('stopwords')

# 全局配置类
class Config:
    YAHOO_TRAIN_PATH = "C:/Users/31278/Desktop/text_classification/yahoo_answers_csv/train.csv"
    YAHOO_TEST_PATH = "C:/Users/31278/Desktop/text_classification/yahoo_answers_csv/test.csv"
    YELP_TRAIN_PATH = "C:/Users/31278/Desktop/text_classification/yelp_review_polarity_csv/train.csv"
    YELP_TEST_PATH = "C:/Users/31278/Desktop/text_classification/yelp_review_polarity_csv/test.csv"
    SAVE_MODEL_PATH = "best_models/"
    SAVE_PLOT_PATH = "model_plots/"  # 新增：图像保存目录
    
    SAMPLE_RATIO = 0.2
    MAX_VOCAB_SIZE = 50000
    YAHOO_MAX_LEN = 300
    YELP_MAX_LEN = 200
    EMBEDDING_DIM = 100
    
    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 0.001
    PATIENCE = 3

# 创建模型和图像保存目录
os.makedirs(Config.SAVE_MODEL_PATH, exist_ok=True)
os.makedirs(Config.SAVE_PLOT_PATH, exist_ok=True)  # 新增：创建图像目录
config = Config()

# ---------------------- 新增：4类核心图像生成函数 ----------------------
def plot_dense_block_structure(save_path):
    """复现DenseNet论文核心图：Dense Block密集连接结构"""
    G = nx.DiGraph()
    nodes = ["Input"] + [f"Layer {i+1}" for i in range(3)]  # 3层Dense Block（和代码一致）
    G.add_nodes_from(nodes)
    
    # 密集连接：每层连接所有前面的层
    for i in range(3):
        for j in range(i+1):
            G.add_edge(nodes[j], nodes[i+1])
    
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw(G, pos, with_labels=True, node_size=4000, node_color="#87CEEB", 
            font_size=12, font_weight="bold", arrows=True, arrowstyle="->", 
            arrowsize=25, edge_color="#696969")
    plt.title("Dense Block Dense Connection Structure (Paper Reproduction)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_vdcnn_structure(save_path):
    """复现VDCNN论文核心图：深层卷积网络结构"""
    layers = [
        "Input\n(Seq Length: 300/200)",
        "Embedding\n(100 Dim)",
        "Conv1D (3×100→64)",
        "Conv1D (3×64→64)",
        "Conv1D (3×64→64)",
        "Conv1D (7×64→64)",
        "Global Max Pooling",
        "Dropout (0.5)",
        "Output Layer\n(10/2 Classes)"
    ]
    
    plt.figure(figsize=(12, 7))
    y_pos = np.linspace(0.8, 0.2, len(layers))
    rect_width = 0.15
    rect_height = 0.08
    
    for i, layer in enumerate(layers):
        # 绘制层矩形
        plt.Rectangle((0.425, y_pos[i]-rect_height/2), rect_width, rect_height, 
                      facecolor="#FFB6C1" if "Conv" in layer else "#98FB98", 
                      edgecolor="black", linewidth=1.5)
        plt.text(0.5, y_pos[i], layer, ha="center", va="center", fontsize=10, font_weight="bold")
        
        # 绘制连接箭头
        if i < len(layers)-1:
            plt.arrow(0.5, y_pos[i]-rect_height/2 - 0.01, 0, 
                      y_pos[i+1]-y_pos[i] + rect_height + 0.02, 
                      head_width=0.02, head_length=0.01, fc="black", ec="black")
    
    plt.xlim(0.3, 0.7)
    plt.ylim(0.1, 0.9)
    plt.axis("off")
    plt.title("Very Deep CNN (VDCNN) Structure (Paper Reproduction)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_training_history(history, model_name, dataset_name, save_path):
    """训练曲线：Loss + Accuracy（覆盖所有模型和数据集）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss曲线
    ax1.plot(history.history['loss'], label=f"Train Loss", linewidth=2, color="#FF6B6B")
    ax1.plot(history.history['val_loss'], label=f"Val Loss", linewidth=2, color="#4ECDC4")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title(f"{model_name} - {dataset_name} Loss Curve", fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy曲线
    ax2.plot(history.history['accuracy'], label=f"Train Accuracy", linewidth=2, color="#FF6B6B")
    ax2.plot(history.history['val_accuracy'], label=f"Val Accuracy", linewidth=2, color="#4ECDC4")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title(f"{model_name} - {dataset_name} Accuracy Curve", fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle(f"Training History: {model_name} on {dataset_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, num_classes, model_name, dataset_name, save_path):
    """混淆矩阵（覆盖所有模型和数据集）"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=[f"Class {i}" for i in range(num_classes)],
                yticklabels=[f"Class {i}" for i in range(num_classes)])
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(f"Confusion Matrix: {model_name} on {dataset_name}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_model_comparison(all_results, save_path):
    """模型性能对比图（VDCNN vs DenseNet，覆盖Yahoo和Yelp）"""
    models = ["Very Deep CNN", "Text DenseNet"]
    datasets = ["Yahoo Answers (10-class)", "Yelp Polarity (2-class)"]
    yahoo_acc = [all_results['vdcnn_yahoo'], all_results['densenet_yahoo']]
    yelp_acc = [all_results['vdcnn_yelp'], all_results['densenet_yelp']]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, yahoo_acc, width, label=datasets[0], color="#FF9999", alpha=0.8)
    rects2 = ax.bar(x + width/2, yelp_acc, width, label=datasets[1], color="#66B2FF", alpha=0.8)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Performance Comparison (VDCNN vs DenseNet)", fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0.5, 1.0)  # 准确率范围（更直观）
    ax.grid(axis="y", alpha=0.3)
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
    
    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_feature_tsne(model, X_test, y_test, num_classes, model_name, dataset_name, save_path):
    """特征可视化（TSNE降维，复现论文特征区分度图）"""
    # 提取全局池化层特征
    feature_layer = model.get_layer("global_pool")
    feature_model = Model(inputs=model.input, outputs=feature_layer.output)
    features = feature_model.predict(X_test[:500], verbose=0)  # 取500个样本避免卡顿
    
    # TSNE降维到2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab10" if num_classes >=10 else "tab2", num_classes)
    for i in range(num_classes):
        mask = y_test[:500] == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                    c=[colors(i)], label=f"Class {i}", alpha=0.7, s=50, edgecolors="black", linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"Feature Visualization (TSNE): {model_name} on {dataset_name}", fontsize=14, pad=20)
    plt.xlabel("TSNE Dimension 1", fontsize=11)
    plt.ylabel("TSNE Dimension 2", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------- 原有代码（未改动） ----------------------
# 数据加载函数
def load_yahoo_data(train_path, test_path):
    train_df = pd.read_csv(train_path, header=None, names=['category', 'title', 'content', 'answer'])
    test_df = pd.read_csv(test_path, header=None, names=['category', 'title', 'content', 'answer'])
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.dropna(subset=['title', 'content', 'answer', 'category'])
    df['text'] = df['title'] + " " + df['content'] + " " + df['answer']
    df_sample, _ = train_test_split(df, test_size=1-config.SAMPLE_RATIO, 
                                    stratify=df['category'], random_state=42)
    label_encoder = LabelEncoder()
    df_sample['label'] = label_encoder.fit_transform(df_sample['category'])
    return df_sample['text'].values, df_sample['label'].values, len(label_encoder.classes_)

def load_yelp_data(train_path, test_path):
    train_df = pd.read_csv(train_path, header=None, names=['label', 'text'])
    test_df = pd.read_csv(test_path, header=None, names=['label', 'text'])
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.dropna(subset=['text'])
    df_sample, _ = train_test_split(df, test_size=1-config.SAMPLE_RATIO, 
                                    stratify=df['label'], random_state=42)
    df_sample['label'] = df_sample['label'].map({1:0, 2:1})
    return df_sample['text'].values, df_sample['label'].values, 2

# 文本预处理函数
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_preprocess(texts, max_len, tokenizer=None, fit_tokenizer=True):
    stop_words = set(stopwords.words('english'))
    tokenized_texts = []
    for text in texts:
        cleaned_text = clean_text(text)
        tokens = word_tokenize(cleaned_text)
        filtered_tokens = [token for token in tokens if token not in stop_words]
        tokenized_texts.append(filtered_tokens)
    
    if fit_tokenizer:
        tokenizer = Tokenizer(num_words=config.MAX_VOCAB_SIZE, oov_token='<OOV>')
        tokenizer.fit_on_texts(tokenized_texts)
    
    sequences = tokenizer.texts_to_sequences(tokenized_texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer

# 模型构建函数
def build_very_deep_cnn(input_len, num_classes):
    inputs = Input(shape=(input_len,), name="input_layer")
    
    embedding = Embedding(
        input_dim=config.MAX_VOCAB_SIZE + 1,
        output_dim=config.EMBEDDING_DIM,
        input_length=input_len,
        trainable=True,
        name="embedding_layer"
    )(inputs)
    
    x = Conv1D(filters=64, kernel_size=3, padding='same', name="conv_3gram")(embedding)
    x = BatchNormalization(name="bn1")(x)
    x = ReLU(name="relu1")(x)
    x = MaxPooling1D(pool_size=2, strides=1, padding='same', name="pool1")(x)
    
    x = Conv1D(filters=64, kernel_size=5, padding='same', name="conv_5gram")(x)
    x = BatchNormalization(name="bn2")(x)
    x = ReLU(name="relu2")(x)
    x = MaxPooling1D(pool_size=2, strides=1, padding='same', name="pool2")(x)
    
    x = Conv1D(filters=64, kernel_size=3, padding='same', name="conv_3gram_2")(x)
    x = BatchNormalization(name="bn3")(x)
    x = ReLU(name="relu3")(x)
    x = MaxPooling1D(pool_size=2, strides=1, padding='same', name="pool3")(x)
    
    x = Conv1D(filters=64, kernel_size=7, padding='same', name="conv_7gram")(x)
    x = BatchNormalization(name="bn4")(x)
    x = ReLU(name="relu4")(x)
    
    x = GlobalMaxPooling1D(name="global_pool")(x)
    x = Dropout(0.5, name="dropout")(x)
    
    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    units = num_classes if num_classes > 2 else 1
    outputs = Dense(units, activation=activation, name="output_layer")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Very_Deep_CNN")
    return model

def dense_block(x, num_layers, growth_rate, block_name):
    features = [x]
    for i in range(num_layers):
        bottleneck = Conv1D(
            filters=4 * growth_rate,
            kernel_size=1,
            padding='same',
            name=f"{block_name}_bottleneck_{i}"
        )(Concatenate(name=f"{block_name}_concat_{i}")(features))
        bottleneck = BatchNormalization(name=f"{block_name}_bn_bottleneck_{i}")(bottleneck)
        bottleneck = ReLU(name=f"{block_name}_relu_bottleneck_{i}")(bottleneck)
        
        conv = Conv1D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            name=f"{block_name}_conv_{i}"
        )(bottleneck)
        conv = BatchNormalization(name=f"{block_name}_bn_conv_{i}")(conv)
        conv = ReLU(name=f"{block_name}_relu_conv_{i}")(conv)
        
        features.append(conv)
    
    return Concatenate(name=f"{block_name}_final_concat")(features)

def transition_layer(x, compression, layer_name):
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

def build_text_densenet(input_len, num_classes, embedding_matrix=None):
    inputs = Input(shape=(input_len,), name="input_layer")
    
    if embedding_matrix is not None:
        embedding = Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            input_length=input_len,
            weights=[embedding_matrix],
            trainable=False,
            name="pretrained_embedding_layer"
        )(inputs)
    else:
        embedding = Embedding(
            input_dim=config.MAX_VOCAB_SIZE + 1,
            output_dim=config.EMBEDDING_DIM,
            input_length=input_len,
            trainable=True,
            name="random_embedding_layer"
        )(inputs)
    
    x = Conv1D(filters=32, kernel_size=3, padding='same', name="init_conv")(embedding)
    x = BatchNormalization(name="init_bn")(x)
    x = ReLU(name="init_relu")(x)
    
    x = dense_block(x, num_layers=3, growth_rate=16, block_name="dense_block1")
    x = transition_layer(x, compression=0.5, layer_name="transition1")
    
    x = dense_block(x, num_layers=3, growth_rate=16, block_name="dense_block2")
    x = transition_layer(x, compression=0.5, layer_name="transition2")
    
    x = GlobalMaxPooling1D(name="global_pool")(x)
    x = Dropout(0.5, name="dropout")(x)
    
    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    outputs = Dense(num_classes, activation=activation, name="output_layer")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Text_DenseNet")
    return model

# 训练与评估函数（修改：返回准确率，用于后续对比）
def train_model(model, X_train, y_train, X_val, y_val, num_classes, model_name):
    loss_fn = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(config.SAVE_MODEL_PATH, f"{model_name}_best.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, lr_scheduler, model_checkpoint],
        shuffle=True
    )
    return history, model  # 新增：返回训练好的模型

def evaluate_model(model, X_test, y_test, num_classes, model_name, dataset_name):
    y_pred_proba = model.predict(X_test, verbose=0)
    if num_classes > 2:
        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='micro', zero_division=0
        )
        print(f"\n【{dataset_name} - {model_name} Multi-class Results】")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Micro F1: {micro_f1:.4f}")
    else:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\n【{dataset_name} - {model_name} Binary-class Results】")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
    
    # 生成混淆矩阵（新增：调用图像函数）
    cm_save_path = os.path.join(config.SAVE_PLOT_PATH, f"{model_name}_{dataset_name}_confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, num_classes, model_name, dataset_name, cm_save_path)
    
    # 生成特征可视化（新增：调用TSNE函数）
    tsne_save_path = os.path.join(config.SAVE_PLOT_PATH, f"{model_name}_{dataset_name}_feature_tsne.png")
    plot_feature_tsne(model, X_test, y_test, num_classes, model_name, dataset_name, tsne_save_path)
    
    return accuracy  # 新增：返回准确率

# ---------------------- 主函数（修改：调用所有图像生成函数） ----------------------
def main():
    # 存储所有模型的准确率（用于后续对比）
    all_results = {}
    
    # 1. 加载数据集
    print("="*50)
    print("1. Loading Yahoo Answers dataset...")
    yahoo_texts, yahoo_labels, yahoo_num_classes = load_yahoo_data(
        config.YAHOO_TRAIN_PATH, config.YAHOO_TEST_PATH
    )
    yahoo_sequences, yahoo_tokenizer = text_preprocess(
        yahoo_texts, max_len=config.YAHOO_MAX_LEN, fit_tokenizer=True
    )
    X_train_yahoo, X_temp_yahoo, y_train_yahoo, y_temp_yahoo = train_test_split(
        yahoo_sequences, yahoo_labels, test_size=0.2, stratify=yahoo_labels, random_state=42
    )
    X_val_yahoo, X_test_yahoo, y_val_yahoo, y_test_yahoo = train_test_split(
        X_temp_yahoo, y_temp_yahoo, test_size=0.5, stratify=y_temp_yahoo, random_state=42
    )
    print(f"Yahoo dataset ready: Train {len(X_train_yahoo)}, Val {len(X_val_yahoo)}, Test {len(X_test_yahoo)}")

    print("\n" + "="*50)
    print("2. Loading Yelp Review Polarity dataset...")
    yelp_texts, yelp_labels, yelp_num_classes = load_yelp_data(
        config.YELP_TRAIN_PATH, config.YELP_TEST_PATH
    )
    yelp_sequences, yelp_tokenizer = text_preprocess(
        yelp_texts, max_len=config.YELP_MAX_LEN, fit_tokenizer=True
    )
    X_train_yelp, X_temp_yelp, y_train_yelp, y_temp_yelp = train_test_split(
        yelp_sequences, yelp_labels, test_size=0.2, stratify=yelp_labels, random_state=42
    )
    X_val_yelp, X_test_yelp, y_val_yelp, y_test_yelp = train_test_split(
        X_temp_yelp, y_temp_yelp, test_size=0.5, stratify=y_temp_yelp, random_state=42
    )
    print(f"Yelp dataset ready: Train {len(X_train_yelp)}, Val {len(X_val_yelp)}, Test {len(X_test_yelp)}")

    # 2. 训练VDCNN（Yahoo）
    print("\n" + "="*50)
    print("3. Training Very Deep CNN (Yahoo dataset)...")
    vdcnn_yahoo = build_very_deep_cnn(
        input_len=config.YAHOO_MAX_LEN,
        num_classes=yahoo_num_classes
    )
    vdcnn_yahoo_history, vdcnn_yahoo_model = train_model(
        vdcnn_yahoo, X_train_yahoo, y_train_yahoo, X_val_yahoo, y_val_yahoo,
        num_classes=yahoo_num_classes, model_name="Very_Deep_CNN_Yahoo"
    )
    # 生成训练曲线
    vdcnn_yahoo_history_path = os.path.join(config.SAVE_PLOT_PATH, "Very_Deep_CNN_Yahoo_training_history.png")
    plot_training_history(vdcnn_yahoo_history, "Very Deep CNN", "Yahoo Answers", vdcnn_yahoo_history_path)
    # 评估并保存准确率
    all_results['vdcnn_yahoo'] = evaluate_model(vdcnn_yahoo_model, X_test_yahoo, y_test_yahoo, yahoo_num_classes, "Very_Deep_CNN", "Yahoo_Answers")

    # 3. 训练VDCNN（Yelp）
    print("\n" + "="*50)
    print("4. Training Very Deep CNN (Yelp dataset)...")
    vdcnn_yelp = build_very_deep_cnn(
        input_len=config.YELP_MAX_LEN,
        num_classes=yelp_num_classes
    )
    vdcnn_yelp_history, vdcnn_yelp_model = train_model(
        vdcnn_yelp, X_train_yelp, y_train_yelp, X_val_yelp, y_val_yelp,
        num_classes=yelp_num_classes, model_name="Very_Deep_CNN_Yelp"
    )
    # 生成训练曲线
    vdcnn_yelp_history_path = os.path.join(config.SAVE_PLOT_PATH, "Very_Deep_CNN_Yelp_training_history.png")
    plot_training_history(vdcnn_yelp_history, "Very Deep CNN", "Yelp Polarity", vdcnn_yelp_history_path)
    # 评估并保存准确率
    all_results['vdcnn_yelp'] = evaluate_model(vdcnn_yelp_model, X_test_yelp, y_test_yelp, yelp_num_classes, "Very_Deep_CNN", "Yelp_Polarity")

    # 4. 训练DenseNet（Yahoo）
    print("\n" + "="*50)
    print("5. Training Text DenseNet (Yahoo dataset)...")
    densenet_yahoo = build_text_densenet(
        input_len=config.YAHOO_MAX_LEN,
        num_classes=yahoo_num_classes
    )
    densenet_yahoo_history, densenet_yahoo_model = train_model(
        densenet_yahoo, X_train_yahoo, y_train_yahoo, X_val_yahoo, y_val_yahoo,
        num_classes=yahoo_num_classes, model_name="Text_DenseNet_Yahoo"
    )
    # 生成训练曲线
    densenet_yahoo_history_path = os.path.join(config.SAVE_PLOT_PATH, "Text_DenseNet_Yahoo_training_history.png")
    plot_training_history(densenet_yahoo_history, "Text DenseNet", "Yahoo Answers", densenet_yahoo_history_path)
    # 评估并保存准确率
    all_results['densenet_yahoo'] = evaluate_model(densenet_yahoo_model, X_test_yahoo, y_test_yahoo, yahoo_num_classes, "Text_DenseNet", "Yahoo_Answers")

    # 5. 训练DenseNet（Yelp）
    print("\n" + "="*50)
    print("6. Training Text DenseNet (Yelp dataset)...")
    densenet_yelp = build_text_densenet(
        input_len=config.YELP_MAX_LEN,
        num_classes=yelp_num_classes
    )
    densenet_yelp_history, densenet_yelp_model = train_model(
        densenet_yelp, X_train_yelp, y_train_yelp, X_val_yelp, y_val_yelp,
        num_classes=yelp_num_classes, model_name="Text_DenseNet_Yelp"
    )
    # 生成训练曲线
    densenet_yelp_history_path = os.path.join(config.SAVE_PLOT_PATH, "Text_DenseNet_Yelp_training_history.png")
    plot_training_history(densenet_yelp_history, "Text DenseNet", "Yelp Polarity", densenet_yelp_history_path)
    # 评估并保存准确率
    all_results['densenet_yelp'] = evaluate_model(densenet_yelp_model, X_test_yelp, y_test_yelp, yelp_num_classes, "Text_DenseNet", "Yelp_Polarity")

    # ---------------------- 新增：生成所有复现图像 ----------------------
    print("\n" + "="*50)
    print("7. Generating paper reproduction plots...")
    
    # 7.1 模型结构示意图（DenseNet + VDCNN）
    dense_block_struct_path = os.path.join(config.SAVE_PLOT_PATH, "DenseNet_Block_Structure.png")
    plot_dense_block_structure(dense_block_struct_path)
    
    vdcnn_struct_path = os.path.join(config.SAVE_PLOT_PATH, "VDCNN_Structure.png")
    plot_vdcnn_structure(vdcnn_struct_path)
    
    # 7.2 模型性能对比图（所有模型+数据集）
    comparison_path = os.path.join(config.SAVE_PLOT_PATH, "Model_Performance_Comparison.png")
    plot_model_comparison(all_results, comparison_path)
    
    print("✅ All plots generated successfully! Saved to: model_plots/")
    print("="*50)

if __name__ == "__main__":
    main()
