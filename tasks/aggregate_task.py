import numpy as np
from albumentations import Compose as A_compose, Resize as A_resize, Normalize as A_Normalize
import tensorflow as tf

from helpers.global_config import Config
from helpers.model_instance_helper import ModelInstanceHelper
from celery_app import celery_app

augment_pipeline_test = A_compose([
    A_resize(Config.INPUT_SHAPE[0], Config.INPUT_SHAPE[1]),
    A_Normalize()
])

def augment_img_test(img,label, transformation):
    if hasattr(img, 'numpy'):
        img = img.numpy()
    if hasattr(label, "numpy"):
        label = label.numpy()
    augment_img = transformation(image = img)
    return augment_img, label

X_test = np.load('./preprocessed/x_test.npy')
Y_test = np.load('./preprocessed/y_test.npy')


@celery_app.task(name='tasks.aggregate_task',bind=True)
def aggregate_task(results):
    all_weights = [r['weights'] for r in results]
    averaged = []
    
    for layer_idx in range(len(all_weights[0])):
        layer_weights = [w[layer_idx] for w in all_weights]
        averaged.append(np.mean(layer_weights, axis=0).tolist())
    
    for r in results:
        print(f"  {r['worker_name']}: Loss={r['loss']:.4f}, Acc={r['accuracy']:.4f}, AUC={r['auc']:.4f}, Recall={r['recall']}, Precision ={r['precision']}")
    
    helper = ModelInstanceHelper()
    model = helper()
    model = helper.set_model_weights(averaged,model)
    
    dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    dataset = dataset.map(
        lambda x,y: tf.py_function(
            func = lambda img, lbl : augment_img_test(img,lbl, augment_pipeline_test),
            inp = [x,y],
            Tout=[tf.float32,tf.float32]
        )
    ).map(lambda x, y: (
        tf.ensure_shape(x,list(Config.INPUT_SHAPE)),
        tf.cast(y, tf.float32)
    )).shuffle(buffer_size=100).batch(3).prefetch(1)
    
    
    history = model.evaluate(dataset,verbose="0")
    
    metrics = {
        "Epoch": results[0]['epoch'],
        "loss": history[0],
        "accuracy": history[1],
        "auc": history[2],
        "recall": history[3],
        "precision": history[4]
    }
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    return {
        "weights": helper.get_model_weights(model)
    }