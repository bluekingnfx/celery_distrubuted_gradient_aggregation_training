from keras.models import load_model
import numpy as np
from pathlib import Path

from helpers.write_to_csv import write_to_csv
from helpers.global_config import Config
from helpers.apply_tensor_transformation import applyTensorTransformation

X_test = np.load('./preprocessed/x_test.npy')
Y_test = np.load('./preprocessed/y_test.npy')

def evaluation_model_on_test_data(epoch_number:int,file_name:str):
    parent_path = Path(__file__).parent.parent
    model_path = parent_path / 'breast_cancer_model.keras'
    
    if model_path.exists() is False:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = load_model(model_path)
    
    dataset = applyTensorTransformation(X_test,Y_test,shuffle_no=100,batch_no=3,train_preset=False)
    
    history = model.evaluate(dataset,verbose="0",return_dict=True) # type: ignore

    metrics: dict[str, float | str]= {
        **history,
        Config.EPOCH: epoch_number,
        Config.FILE_NAME_CONVENTION: file_name
    }
    
    write_to_csv(metrics,evaluation_metrics=True)
    