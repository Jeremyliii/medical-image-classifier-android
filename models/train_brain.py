"""
脑部MRI肿瘤检测模型训练
数据集：Brain MRI Images for Brain Tumor Detection
Kaggle: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("🧠 脑部MRI肿瘤检测模型训练程序")
print("=" * 60)

# ========== 1. 配置参数 ==========
print("\n[1/8] 配置训练参数...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录
DATA_DIR = f'{BASE_DIR}/datasets/brain_tumor'
MODEL_DIR = f'{BASE_DIR}/models'
LABELS_DIR = f'{BASE_DIR}/labels'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# 检查数据集
if not os.path.exists(DATA_DIR):
    print(f"❌ 错误: 找不到数据集 {DATA_DIR}")
    print("请确保已下载并解压脑肿瘤数据集到指定位置")
    print("下载地址: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
    print("\n数据集结构应为:")
    print("  brain_tumor/")
    print("  ├── yes/  (有肿瘤)")
    print("  └── no/   (无肿瘤)")
    exit(1)

# 训练参数
IMG_SIZE = 224
BATCH_SIZE = 8  # 很小的batch因为数据集很小(253张)
EPOCHS = 40
LEARNING_RATE = 0.0003
MODEL_NAME = 'brain'
CLASS_NAMES = ['NO_TUMOR', 'TUMOR']
CLASS_NAMES_CN = ['无肿瘤', '有肿瘤']

print(f"  数据路径: {DATA_DIR}")
print(f"  图像大小: {IMG_SIZE}x{IMG_SIZE}")
print(f"  批次大小: {BATCH_SIZE}")
print(f"  训练轮数: {EPOCHS}")
print(f"  模型名称: {MODEL_NAME}")

# ========== 2. 加载数据 ==========
print("\n[2/8] 加载数据...")

# 数据集较小，使用20%验证
train_ds = keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    color_mode='grayscale',  # MRI灰度图
    seed=42,
    validation_split=0.2,
    subset='training'
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    color_mode='grayscale',
    seed=42,
    validation_split=0.2,
    subset='validation'
)

print(f"  类别: {train_ds.class_names}")
print(f"  训练批次: {len(train_ds)}")
print(f"  验证批次: {len(val_ds)}")
print("  ⚠️ 注意: 数据集较小(253张)，已启用强数据增强")

# ========== 3. 数据预处理 ==========
print("\n[3/8] 配置数据处理...")

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment(image, label):
    # 强数据增强（因为数据集小）
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    # 随机裁剪增强
    padded = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 30, IMG_SIZE + 30)
    image = tf.image.random_crop(padded, [IMG_SIZE, IMG_SIZE, 1])
    
    return image, label

train_ds = train_ds.map(preprocess).map(augment)
test_ds = val_ds.map(preprocess)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(200).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("  ✓ 强数据增强已配置（翻转、亮度、对比度、裁剪）")
print("  ✓ 数据管道已优化")

# ========== 4. 构建模型 ==========
print("\n[4/8] 构建模型...")

# 使用MobileNetV2（轻量但效果好）
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    
    # 灰度转RGB
    layers.Conv2D(3, (1, 1), padding='same'),
    
    # 内置数据增强
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    
    # 预训练模型
    base_model,
    
    # 分类头（针对小数据集优化）
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # 较高dropout防过拟合
    
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)),
    layers.Dropout(0.3),
    
    layers.Dense(1, activation='sigmoid')
])

print(f"  基础模型: MobileNetV2")
print(f"  模型层数: {len(model.layers)}")
print(f"  可训练参数: {model.count_params():,}")

# ========== 5. 编译模型 ==========
print("\n[5/8] 编译模型...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

print("  ✓ 优化器: Adam (lr=0.0003)")
print("  ✓ 损失函数: Binary Crossentropy")
print("  ✓ 正则化: L2(0.02) + Dropout(0.5)")

# ========== 6. 训练模型 ==========
print("\n[6/8] 开始训练...\n")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        f'{MODEL_DIR}/{MODEL_NAME}_best.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ========== 7. 评估和保存 ==========
print("\n[7/8] 评估模型...")

results = model.evaluate(test_ds, verbose=0)

print("\n" + "=" * 60)
print(f"🧠 {MODEL_NAME.upper()} 训练完成！最终测试结果:")
print("=" * 60)
print(f"  损失值:   {results[0]:.4f}")
print(f"  准确率:   {results[1]*100:.2f}%")
print(f"  精确率:   {results[2]*100:.2f}%")
print(f"  召回率:   {results[3]*100:.2f}%")
print(f"  AUC:      {results[4]:.4f}")
print("=" * 60)

# 保存模型
model_path = f'{MODEL_DIR}/{MODEL_NAME}_model.h5'
model.save(model_path)
print(f"\n✅ 模型已保存: {model_path}")

# 保存标签
labels_path = f'{LABELS_DIR}/{MODEL_NAME}_labels.txt'
with open(labels_path, 'w', encoding='utf-8') as f:
    for en, cn in zip(CLASS_NAMES, CLASS_NAMES_CN):
        f.write(f"{en}|{cn}\n")
print(f"✅ 标签已保存: {labels_path}")

# ========== 8. 绘制训练曲线 ==========
print("\n[8/8] 绘制训练曲线...")

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'🧠 {MODEL_NAME.upper()} 脑肿瘤检测模型训练曲线', fontsize=16, fontweight='bold')

axes[0, 0].plot(history.history['accuracy'], 'b-', label='训练', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], 'r-', label='验证', linewidth=2)
axes[0, 0].set_title('准确率', fontsize=14)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history.history['loss'], 'b-', label='训练', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], 'r-', label='验证', linewidth=2)
axes[0, 1].set_title('损失值', fontsize=14)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history.history['precision'], 'g-', label='训练', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], 'orange', label='验证', linewidth=2)
axes[1, 0].set_title('精确率', fontsize=14)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history.history['auc'], 'purple', label='训练', linewidth=2)
axes[1, 1].plot(history.history['val_auc'], 'brown', label='验证', linewidth=2)
axes[1, 1].set_title('AUC', fontsize=14)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('AUC')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/{MODEL_NAME}_training.png', dpi=150, bbox_inches='tight')
print(f"✅ 训练曲线已保存: {MODEL_NAME}_training.png")

print("\n" + "=" * 60)
print(f"🎉 {MODEL_NAME.upper()} 脑肿瘤检测模型训练全部完成！")
print("=" * 60)
