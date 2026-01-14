from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten
from keras.metrics import Precision, Recall, AUC, F1Score
from keras.optimizers import Adam
import numpy as np

from helpers.global_config import Config

class ModelInstanceHelper:
    def __init__(self, num_workers = None):
        self.nw = num_workers or Config.num_workers
    def __call__(self):
        model = Sequential([
            Input(shape=(*Config.INPUT_SHAPE, Config.INPUT_DIM)),
            
            Conv2D(filters=128, kernel_size=3, activation="relu"),
            MaxPooling2D((2,2)),
            BatchNormalization(),
            Dropout(0.2),
            
            Conv2D(filters = 64, kernel_size=(3,3), activation="relu"),
            MaxPooling2D((2,2)),
            BatchNormalization(),
            Dropout(0.1),
            
            Conv2D(filters = 32, kernel_size=(2,2), activation='relu'),
            BatchNormalization(), 
            Flatten(),       
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        scaled_learning_rate = self.nw*Config.BASE_LR
        
        opt = Adam(learning_rate=scaled_learning_rate)
        
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics = [Config.ACCURACY, Precision(name=Config.PRECISION), Recall(name=Config.RECALL), AUC(name=Config.AUC), F1Score(name=Config.F1_SCORE)]) # type: ignore
        
        return model 
    
    def get_model_weights(self,model):
        learned_weights = [w.tolist() for w in model.get_weights()]
        return learned_weights
    
    def set_model_weights(self,weights_list,model):
        model.set_weights([np.array(w) for w in weights_list])
        return model
    