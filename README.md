# MedicalAI：手机上离线跑的医学图像识别

一个 Android 应用加四个 TensorFlow Lite 模型：选一张医学图像，在手机本地（不联网）判断肺炎、乳腺肿瘤良恶性、脑肿瘤和疟疾感染。仓库里有训练脚本、转换脚本、数据集、打包好的 APK 和用来复现准确率的评估脚本。

An Android app with four on-device TFLite classifiers (pneumonia X-ray, breast ultrasound, brain MRI, malaria blood cells), transfer-learned from MobileNetV2 / ResNet50V2. Training, conversion and evaluation scripts are included; measured results are below.

> 仅用于学习和研究，**不能用于临床诊断**。

## 四个模型和实测结果

结果是用 **App 里实际打包的 `.tflite` 模型**，在训练时没参与训练的数据上跑出来的（`evaluation/evaluate_tflite.py`，2026-11 重新测）：

| 任务 | 数据 | 骨干网络 | 输入 | 测试图片数 | 准确率 | AUC | 查出率（敏感度） | 正常判对率（特异度） |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 肺炎（胸部 X 光） | [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 自带 test 集 | MobileNetV2（冻结） | 224 灰度 | 624 | 84.9% | 0.956 | 98.7% | 61.9% |
| 乳腺肿瘤良恶性（超声） | Breast Ultrasound Images，25% 验证集 | ResNet50V2（微调后 20 层） | 224 RGB | 161 | 85.1% | 0.912 | 65.5% | 95.3% |
| 脑肿瘤（MRI） | Brain MRI Images for Brain Tumor，20% 验证集 | MobileNetV2（冻结） | 224 灰度 | 50 | 80.0% | 0.922 | 69.7% | 100% |
| 疟疾（血细胞） | [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)，20% 验证集 | MobileNetV2（微调后 30 层） | 150 RGB | 5511 | 95.4% | 0.987 | 92.7% | 98.0% |

怎么读这张表：

- **肺炎模型偏向“报病”**：真肺炎几乎都能查出来（390 张漏 5 张），但 234 张正常片里有 89 张被说成肺炎。做初筛可以接受，做诊断不行。原因之一是训练集里肺炎图是正常图的近 3 倍，训练时没加类别权重。
- **乳腺和脑肿瘤模型偏向“说没事”**：特异度高，但恶性肿瘤漏掉了 35%（55 张漏 19 张），脑肿瘤漏了 30%。对医疗场景来说，漏诊比误报更危险，这两个模型目前的阈值（0.5）并不合适。AUC 都在 0.9 以上，说明调低阈值能换来更高的查出率。
- **脑肿瘤只有 50 张测试图**，数字波动会很大。
- **这些数字偏乐观**：训练脚本把同一份验证 / 测试集用来挑最好的那一轮（EarlyStopping / ModelCheckpoint），没有再留一份完全没碰过的数据。

## 实现

- **迁移学习**：ImageNet 预训练的 MobileNetV2 / ResNet50V2 做骨干，上面接全局平均池化 + 两层全连接（带 Dropout，乳腺和脑肿瘤还加了 L2 正则）+ 一个 sigmoid 输出。灰度图前面加一层 1×1 卷积，把 1 个通道映射成骨干需要的 3 个通道。
- **训练**（`models/train_*.py`）：Keras `image_dataset_from_directory` 读数据、像素除以 255；随机翻转、亮度、对比度做数据增强（脑肿瘤数据少，另加旋转、缩放、随机裁剪）；Adam；EarlyStopping、ReduceLROnPlateau；监控 AUC 或准确率。
- **转换**（`models/convert_all.py`）：`.h5` → `.tflite`，同时生成 `models_config.txt`。
- **Android 端**（`app/MainActivity.java`）：从 assets 载入 `.tflite`，图片缩放到模型的输入尺寸、按同样方式归一化（灰度取 RGB 平均），用 TFLite `Interpreter` 推理，输出 > 0.5 判为第二类，显示中英文结果和置信度。四个模型在界面上切换。

## 目录

```text
app/          MainActivity.java、activity_main.xml、app-apk.apk（可直接安装）
models/       训练脚本 train_*.py、转换脚本 convert_all.py、四个 .tflite 模型
labels/       每个模型的标签（顺序 = 数据集文件夹的字母顺序）
datasets/     四个数据集（约 1.8 GB）
evaluation/   evaluate_tflite.py（复现上表）、results.json（含混淆矩阵）
```

## 运行

```bash
pip install tensorflow scikit-learn

# 复现上面的结果
python evaluation/evaluate_tflite.py pneumonia breast brain malaria

# 重新训练、转换（路径按仓库位置自动计算）
python models/train_all.py          # 或 train_pneumonia.py 等单独训练
python models/convert_all.py
```

Android：把 `app/app-apk.apk` 装到手机或模拟器即可。

## 这次整理改了什么

- 训练好的 `.tflite` 模型原来只在 APK 里，现在放进了 `models/`。
- 疟疾模型的输入是 **150×150**，原文档和 `convert_all.py` 写成了 128；标签文件 `labels/malaria_labels.txt` 的顺序是反的（App 里已经改对了，文件没改），都已修正。
- 训练 / 转换脚本不再写死 `C:/MedicalAI`，按仓库位置找数据。
- 加了评估脚本和上面的实测结果。

## 可以继续改进

- 训练时加类别权重，或按 AUC 曲线给每个模型单独选阈值（乳腺、脑肿瘤应优先提高查出率）。
- 留出一份只在最后用一次的测试集，让报告的数字不偏乐观。
- 脑肿瘤数据太少（253 张），换更大的数据集。
