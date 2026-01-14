import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from helpers.global_config import Config
from helpers.model_instance_helper import ModelInstanceHelper
from celery_app import celery_app
from helpers.write_to_csv import write_to_csv


X_test = np.load('./preprocessed/x_test.npy')
Y_test = np.load('./preprocessed/y_test.npy')

def flatten_values(results,y_values_property_name):
    all_batch_preds = [np.array(r[y_values_property_name]) for r in results]
    y_column = np.concatenate(all_batch_preds, axis=0)
    return y_column.flatten()

@celery_app.task(name='tasks.aggregate_task',bind=True)
def aggregate_task(self, results):
    
    worker_name = self.request.hostname
    
    all_weights = [r['weights'] for r in results]
    averaged = []
    
    for layer_idx in range(len(all_weights[0])):
        layer_weights = [w[layer_idx] for w in all_weights]
        averaged.append(np.mean(layer_weights, axis=0).tolist())
    
    helper = ModelInstanceHelper()
    model = helper()
    model = helper.set_model_weights(averaged,model)
    
    
    
    y_vals_pred = flatten_values(results, "y_val_predictions")
    
    y_vals_pred_proba = flatten_values(results, "y_val_predictions_proba")
    
    y_true_vals = flatten_values(results, 'y_val_true')
    
    
    accuracy = accuracy_score(y_true_vals, y_vals_pred)
    precision = precision_score(y_true_vals, y_vals_pred)
    recall = recall_score(y_true_vals, y_vals_pred)
    f1 = f1_score(y_true_vals, y_vals_pred)
    roc_auc = roc_auc_score(y_true_vals, y_vals_pred_proba)
    loss = np.mean([r[Config.LOSS] for r in results])
    
    
    metrics: dict[str, float | str] = {
        Config.ACCURACY: accuracy,
        Config.PRECISION: precision,
        Config.RECALL: recall,
        Config.F1_SCORE: f1,
        Config.AUC: roc_auc,
        Config.EPOCH: np.astype(results[0][Config.EPOCH], dtype=np.float32), # type: ignore
        Config.FILE_NAME_CONVENTION: results[0][Config.FILE_NAME_CONVENTION],
        Config.LOSS: loss
    }
    
    write_to_csv(metrics,evaluation_metrics=False)
    
    return {
        "weights": helper.get_model_weights(model),
        "epoch": results[0][Config.EPOCH],
        "worker_name": worker_name,
    }