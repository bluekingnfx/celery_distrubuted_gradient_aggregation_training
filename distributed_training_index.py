
import numpy as np
from celery import chord


from celery_app import celery_app

if celery_app is None:
    print('Celery client is not initialized')

from helpers.global_config import Config
from celery_app import celery_app
from helpers.model_instance_helper import ModelInstanceHelper
from tasks.train_task import train_task
from tasks.aggregate_task import aggregate_task
from helpers.produce_date_str import create_file_name
from helpers.evaluation_model_on_test_data import evaluation_model_on_test_data


def start_distributed_training():
    X_TRAIN = np.load('preprocessed/x_train.npy')
    file_name = create_file_name() 
    helper = ModelInstanceHelper()
    model = helper()
    model_weights = helper.get_model_weights(model)
    train_len = len(X_TRAIN)
    batch_size = int(train_len/Config.num_workers)
    for epoch in range(Config.EPOCHS):
        tasks = []
        for worker in range(0,Config.num_workers):
            start_ind = batch_size*worker
            end_ind = start_ind + batch_size if worker < Config.num_workers - 1 else train_len
            
            tasks.append(train_task.s( # type: ignore
                start_idx=start_ind,
                end_idx= end_ind,
                initial_weights=model_weights,
                epoch_number=epoch,
                worker_number=worker,
                file_name=file_name
            ))
        
        aggregate_result = chord(tasks)(aggregate_task.s()) # type: ignore
        model_weights = aggregate_result.get()['weights'] #type: ignore
    
    final_model = ModelInstanceHelper()()
    final_model.set_weights([np.array(w) for w in model_weights])
    final_model.save('breast_cancer_model.keras')
    
    print("Best model is saved!")
    print("Evaluating the final model on test data...")
    
    evaluation_model_on_test_data(epoch_number=-1, file_name=file_name)
    
    print('Evaluation completed!')

if __name__ == "__main__":
    start_distributed_training()
