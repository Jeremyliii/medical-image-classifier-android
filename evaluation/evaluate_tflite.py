"""用 App 里实际打包的 TFLite 模型，在训练时没用来训练的数据上算准确率。
用法：python evaluation/evaluate_tflite.py pneumonia breast brain malaria
肺炎用数据集自带的 test 目录；其余三个用训练脚本同样的 validation_split 和 seed=42 切出的验证集。"""
import os, sys, json
import numpy as np, tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'datasets')
CFG = {  # model, size, mode, data dir, split
 'pneumonia': (224, 'grayscale', f'{D}/chest_xray/test', None),
 'breast': (224, 'rgb', f'{D}/breast_ultrasound', .25),
 'brain': (224, 'grayscale', f'{D}/brain_tumor', .2),
 'malaria': (150, 'rgb', f'{D}/malaria', .2),
}
out = {}
for name in sys.argv[1:]:
    size, mode, path, split = CFG[name]
    kw = dict(image_size=(size, size), batch_size=64, label_mode='binary', color_mode=mode, shuffle=False if split is None else True, seed=42)
    if split: kw.update(validation_split=split, subset='validation')
    ds = tf.keras.preprocessing.image_dataset_from_directory(path, **kw)
    it = tf.lite.Interpreter(model_path=os.path.join(ROOT, 'models', f'{name}_model.tflite')); it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    it.resize_tensor_input(inp['index'], [64] + list(inp['shape'][1:])); it.allocate_tensors()
    ys, ps = [], []
    for x, y in ds:
        x = (x.numpy() / 255.0).astype(np.float32); n = len(x)
        if n < 64: x = np.concatenate([x, np.zeros((64 - n,) + x.shape[1:], np.float32)])
        it.set_tensor(inp['index'], x); it.invoke(); p = it.get_tensor(outp['index'])[:n, 0]
        ys += list(y.numpy()[:, 0]); ps += list(p)
    ys, ps = np.array(ys), np.array(ps); pred = (ps > .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(ys, pred).ravel()
    out[name] = dict(classes=ds.class_names, n=len(ys), acc=round(float((pred == ys).mean()), 4), auc=round(float(roc_auc_score(ys, ps)), 4),
                     sens=round(tp / (tp + fn), 4), spec=round(tn / (tn + fp), 4), cm=[[int(tn), int(fp)], [int(fn), int(tp)]])
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(ROOT, 'evaluation', 'results.json'), 'w'), indent=1, default=float)
