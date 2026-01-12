
import numpy as np
import tensorflow as tf

from helpers.global_config import Config
from celery_app import celery_app
from helpers.model_instance_helper import ModelInstanceHelper
from preprocessing.augment_images import augment_images,augment_pipeline_train

X_train = np.load('preprocessed/x_train.npy')
Y_train = np.load('preprocessed/y_train.npy')


@celery_app.task(celery_app = "tasks.train_task", bind=True)
def train_task(self, start_idx,end_idx, initial_weights, epoch_number, worker_number):
    worker_name = self.request.hostname
    model = ModelInstanceHelper()()
    model = ModelInstanceHelper().set_model_weights(initial_weights, model)
    
    x_batch = X_train[start_idx:end_idx]
    y_batch = Y_train[start_idx:end_idx]
    
    dataset = tf.data.Dataset.from_tensor_slices((x_batch,y_batch))
    
    dataset = dataset.map(
        lambda x,y: tf.py_function(
            func = lambda img,label : augment_images(img,label, augment_pipeline_train),
            inp = [x,y],
            Tout=[tf.float32,tf.float32]
        )
    ).map(lambda img, label: (
        tf.ensure_shape(img, list(Config.INPUT_SHAPE)),
        tf.cast(label, tf.float32)
    )).shuffle(200).batch(2).prefetch(1)
    
    history = model.fit(
        dataset,
        epochs = 1,
        verbose = "0"
    )
    
    weights = ModelInstanceHelper().get_model_weights(model)
    
    return {
        "weights": weights,
        "loss": float(history.history['loss'][0]),
        "accuracy": float(history.history['accuracy'][0]),
        "recall": float(history.history['recall'][0]),
        "auc": float(history.history['auc'][0]),
        "precision":float(history.history['precision'][0]),
        "epoch":  epoch_number,
        "worker_id": worker_number,
        "worker_name": worker_name
    }
