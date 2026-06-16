# MedicalAI

MedicalAI 是一个医学图像识别项目，包含用于 Android 端部署的 TensorFlow Lite 模型、标签文件、训练脚本、数据集目录以及已打包的 APK。

项目目前支持 4 类医学图像识别任务：

| 模型 | 输入尺寸 | 输入类型 | 结果标签 |
| --- | ---: | --- | --- |
| 肺炎检测 | 224 x 224 | grayscale | 正常 / 肺炎 |
| 乳腺癌检测 | 224 x 224 | rgb | 良性 / 恶性 |
| 脑肿瘤检测 | 224 x 224 | grayscale | 无肿瘤 / 有肿瘤 |
| 疟疾检测 | 128 x 128 | rgb | 未感染 / 感染疟疾 |

## 目录结构

```text
C:/MedicalAI
├── app-apk.apk              # Android 应用安装包
├── datasets/                # 训练数据集
│   ├── brain_tumor/
│   ├── breast_ultrasound/
│   ├── chest_xray/
│   └── malaria/
├── labels/                  # Android 推理结果标签
│   ├── brain_labels.txt
│   ├── breast_labels.txt
│   ├── malaria_labels.txt
│   └── pneumonia_labels.txt
└── models/                  # 训练、转换脚本和模型文件
    ├── train_all.py
    ├── train_brain.py
    ├── train_breast.py
    ├── train_malaria.py
    ├── train_pneumonia.py
    ├── convert_all.py
    ├── models_config.txt
    ├── *_model.h5
    ├── *_best.h5
    └── *_model.tflite
```

## 快速使用

### 1. 安装 APK

将 `app-apk.apk` 复制到 Android 手机或模拟器中安装。应用使用 `models/` 中的 `.tflite` 模型和 `labels/` 中的标签文件进行医学图像分类。

### 2. 查看模型配置

模型配置位于：

```text
models/models_config.txt
```

配置格式：

```text
model_key|tflite_file|input_size|input_type|name_cn
```

示例：

```text
pneumonia|pneumonia_model.tflite|224|grayscale|肺炎检测
malaria|malaria_model.tflite|128|rgb|疟疾检测
```

## 训练模型

进入模型目录：

```powershell
cd C:/MedicalAI/models
```

训练全部模型：

```powershell
python train_all.py
```

也可以单独训练某个模型：

```powershell
python train_pneumonia.py
python train_breast.py
python train_brain.py
python train_malaria.py
```

训练脚本会读取 `C:/MedicalAI/datasets/` 下对应的数据集目录。

## 转换为 TensorFlow Lite

训练得到 `.h5` 模型后，运行批量转换脚本：

```powershell
cd C:/MedicalAI/models
python convert_all.py
```

转换完成后会生成或更新：

- `pneumonia_model.tflite`
- `breast_model.tflite`
- `brain_model.tflite`
- `malaria_model.tflite`
- `models_config.txt`

## 标签文件

Android 端展示结果时使用 `labels/` 目录中的标签文件：

```text
pneumonia_labels.txt  NORMAL / PNEUMONIA
breast_labels.txt     BENIGN / MALIGNANT
brain_labels.txt      NO_TUMOR / TUMOR
malaria_labels.txt    UNINFECTED / PARASITIZED
```

每行格式为：

```text
label_key|中文显示名称
```

## 运行环境

训练和转换脚本主要依赖：

- Python 3
- TensorFlow / Keras

可按需要安装：

```powershell
pip install tensorflow
```

## 注意事项

- 当前脚本中的基础路径为 `C:/MedicalAI`，如果移动项目目录，需要同步修改训练和转换脚本中的 `BASE_DIR`。
- 数据集、`.h5` 模型和 `.tflite` 模型文件体积较大，上传到 GitHub 前建议确认仓库容量和 Git LFS 策略。
- 医学图像识别结果仅适合作为辅助参考，不能替代专业医生诊断。
