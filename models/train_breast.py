"""
乳腺超声图像癌症检测模型训练 - 优化简化版（无绘图）
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

print("=" * 60)
print("🎀 乳腺超声癌症检测模型训练程序")
print("=" * 60)

# ========== 配置 ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录
DATA_DIR = f'{BASE_DIR}/datasets/breast_ultrasound'
MODEL_DIR = f'{BASE_DIR}/models'
LABELS_DIR = f'{BASE_DIR}/labels'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

if not os.path.exists(DATA_DIR):
    print(f"❌ 找不到数据集 {DATA_DIR}")
    exit(1)

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0001
MODEL_NAME = 'breast'

print(f"  图像大小: {IMG_SIZE}, 批次: {BATCH_SIZE}, 轮数: {EPOCHS}")

# ========== 加载数据 ==========
print("\n[1/4] 加载数据...")

train_ds = keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    color_mode='rgb',
    seed=42,
    validation_split=0.25,
    subset='training'
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    color_mode='rgb',
    seed=42,
    validation_split=0.25,
    subset='validation'
)

print(f"  类别: {train_ds.class_names}")
print(f"  训练: {len(train_ds)} 批, 验证: {len(val_ds)} 批")

# ========== 数据处理 ==========
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(preprocess).map(augment).cache().shuffle(500).prefetch(AUTOTUNE)
test_ds = val_ds.map(preprocess).cache().prefetch(AUTOTUNE)

# ========== 构建模型 ==========
print("\n[2/4] 构建模型...")

base_model = keras.applications.ResNet50V2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# 微调最后20层
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

print(f"  基础模型: ResNet50V2 (微调最后20层)")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

# ========== 训练 ==========
print("\n[3/4] 开始训练...\n")

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True, mode='max'),
    keras.callbacks.ModelCheckpoint(f'{MODEL_DIR}/{MODEL_NAME}_best.h5', monitor='val_auc', save_best_only=True, mode='max'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
]

model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

# ========== 评估保存 ==========
print("\n[4/4] 评估保存...")

results = model.evaluate(test_ds, verbose=0)

print("\n" + "=" * 60)
print(f"🎀 训练完成！")
print("=" * 60)
print(f"  准确率: {results[1]*100:.2f}%")
print(f"  精确率: {results[2]*100:.2f}%")
print(f"  召回率: {results[3]*100:.2f}%")
print(f"  AUC:    {results[4]:.4f}")
print("=" * 60)

model.save(f'{MODEL_DIR}/{MODEL_NAME}_model.h5')
print(f"✅ 模型已保存")

with open(f'{LABELS_DIR}/{MODEL_NAME}_labels.txt', 'w', encoding='utf-8') as f:
    f.write("BENIGN|良性\nMALIGNANT|恶性\n")
print(f"✅ 标签已保存")