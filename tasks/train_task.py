
import numpy as np

from celery_app import celery_app
from helpers.model_instance_helper import ModelInstanceHelper
from helpers.apply_tensor_transformation import applyTensorTransformation, applyTensorTransformationForX_VAL
from helpers.global_config import Config

X_train = np.load('preprocessed/x_train.npy')
Y_train = np.load('preprocessed/y_train.npy')

X_Val = np.load('preprocessed/x_val.npy')
Y_Val = np.load('preprocessed/y_val.npy')


@celery_app.task(celery_app = "tasks.train_task", bind=True)
def train_task(self, start_idx,end_idx, initial_weights, epoch_number, worker_number, file_name):
    worker_name = self.request.hostname
    helper = ModelInstanceHelper()
    model = helper()
    model = helper.set_model_weights(initial_weights, model)
    
    x_batch = X_train[start_idx:end_idx]
    y_batch = Y_train[start_idx:end_idx]
    
    dataset = applyTensorTransformation(x_batch,y_batch)
    
    history = model.fit(
        dataset,
        epochs = 1,
        verbose = "0",
    )
    
    X_Val_dataset, Y_Val_true = applyTensorTransformationForX_VAL(X_Val, Y_Val, batch_no=Config.TRAIN_BATCH_SIZE)
    
    predictions_proba = model.predict(X_Val_dataset, verbose="0").flatten()
    
    predictions = (predictions_proba > 0.5).astype(int)
    
    weights = helper.get_model_weights(model)
    
    return {
        "y_val_true": Y_Val_true.tolist(),
        Config.EPOCH: epoch_number,
        "y_val_predictions": predictions.tolist(),
        "y_val_predictions_proba": predictions_proba.tolist(),
        "weights": weights,
        "worker_id": worker_number,
        "worker_name": worker_name,
        Config.LOSS: float(history.history[Config.LOSS][0]),
        Config.FILE_NAME_CONVENTION: file_name,
    }
