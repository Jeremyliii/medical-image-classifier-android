"""
批量模型转换脚本
将所有Keras模型转换为TensorFlow Lite格式
"""

import tensorflow as tf
import os

print("=" * 60)
print("🔄 批量TensorFlow Lite模型转换程序")
print("=" * 60)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录
MODEL_DIR = f'{BASE_DIR}/models'

# 定义所有模型信息
MODELS = {
    'pneumonia': {
        'h5_file': 'pneumonia_model.h5',
        'tflite_file': 'pneumonia_model.tflite',
        'name': '肺炎检测',
        'input_type': 'grayscale',
        'input_size': 224
    },
    'breast': {
        'h5_file': 'breast_model.h5',
        'tflite_file': 'breast_model.tflite',
        'name': '乳腺癌检测',
        'input_type': 'rgb',
        'input_size': 224
    },
    'brain': {
        'h5_file': 'brain_model.h5',
        'tflite_file': 'brain_model.tflite',
        'name': '脑肿瘤检测',
        'input_type': 'grayscale',
        'input_size': 224
    },
    'malaria': {
        'h5_file': 'malaria_model.h5',
        'tflite_file': 'malaria_model.tflite',
        'name': '疟疾检测',
        'input_type': 'rgb',
        'input_size': 150
    }
}

def convert_model(model_key, model_info):
    """转换单个模型"""
    h5_path = f'{MODEL_DIR}/{model_info["h5_file"]}'
    tflite_path = f'{MODEL_DIR}/{model_info["tflite_file"]}'
    
    print(f"\n{'='*50}")
    print(f"📦 转换: {model_info['name']}")
    print(f"{'='*50}")
    
    # 检查源文件
    if not os.path.exists(h5_path):
        print(f"  ⏭️  跳过: 找不到 {h5_path}")
        return False
    
    try:
        # 1. 加载模型
        print(f"  [1/4] 加载模型...")
        model = tf.keras.models.load_model(h5_path)
        print(f"        输入: {model.input_shape}")
        print(f"        输出: {model.output_shape}")
        
        # 2. 配置转换器
        print(f"  [2/4] 配置转换器...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # 优化设置
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # 3. 执行转换
        print(f"  [3/4] 执行转换...")
        tflite_model = converter.convert()
        
        # 4. 保存模型
        print(f"  [4/4] 保存模型...")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # 统计信息
        original_size = os.path.getsize(h5_path) / (1024 * 1024)
        tflite_size = len(tflite_model) / (1024 * 1024)
        compression = (1 - tflite_size / original_size) * 100
        
        print(f"\n  ✅ 转换成功!")
        print(f"     原始大小: {original_size:.2f} MB")
        print(f"     TFLite:   {tflite_size:.2f} MB")
        print(f"     压缩率:   {compression:.1f}%")
        
        # 验证模型
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"     验证输入: {input_details[0]['shape']}")
        print(f"     验证输出: {output_details[0]['shape']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 转换失败: {e}")
        return False

def main():
    """主函数"""
    print(f"\n检测到 {len(MODELS)} 个模型配置")
    print(f"模型目录: {MODEL_DIR}")
    
    # 检查目录
    if not os.path.exists(MODEL_DIR):
        print(f"\n❌ 错误: 模型目录不存在 {MODEL_DIR}")
        return
    
    # 转换统计
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    results = []
    
    for key, info in MODELS.items():
        h5_path = f'{MODEL_DIR}/{info["h5_file"]}'
        if not os.path.exists(h5_path):
            skipped_count += 1
            results.append((info['name'], '跳过', '模型文件不存在'))
        elif convert_model(key, info):
            success_count += 1
            results.append((info['name'], '成功', info['tflite_file']))
        else:
            failed_count += 1
            results.append((info['name'], '失败', '转换出错'))
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("📊 转换汇总")
    print("=" * 60)
    
    for name, status, detail in results:
        emoji = '✅' if status == '成功' else ('⏭️' if status == '跳过' else '❌')
        print(f"  {emoji} {name}: {status} ({detail})")
    
    print(f"\n统计:")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  跳过: {skipped_count}")
    
    # 生成模型信息文件
    if success_count > 0:
        print("\n" + "=" * 60)
        print("📝 生成模型配置文件...")
        
        config_path = f'{MODEL_DIR}/models_config.txt'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("# 医学图像识别模型配置\n")
            f.write("# 格式: model_key|tflite_file|input_size|input_type|name_cn\n\n")
            
            for key, info in MODELS.items():
                tflite_path = f'{MODEL_DIR}/{info["tflite_file"]}'
                if os.path.exists(tflite_path):
                    f.write(f"{key}|{info['tflite_file']}|{info['input_size']}|{info['input_type']}|{info['name']}\n")
        
        print(f"  ✅ 配置文件: {config_path}")
    
    print("\n" + "=" * 60)
    print("🎉 批量转换完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 将 models/ 目录下的 .tflite 文件复制到 Android 项目")
    print("  2. 将 labels/ 目录下的 .txt 文件复制到 Android 项目")
    print("  3. 更新 Android 代码以支持多模型选择")

if __name__ == '__main__':
    main()
