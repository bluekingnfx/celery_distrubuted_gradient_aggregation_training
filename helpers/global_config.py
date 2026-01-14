


class Config:
    INPUT_SHAPE = (128,128)
    num_workers = 3
    INPUT_DIM = 3
    BASE_LR = 0.001
    CANCEROUS = 1
    NON_CANCEROUS = 0
    EPOCHS = 10 
    F1_SCORE = "f1_score"
    RECALL = "recall"
    PRECISION = 'precision'
    AUC = 'roc_auc'
    ACCURACY = 'accuracy'
    LOSS = 'loss'
    TRAIN_BATCH_SIZE = 2
    TEST_BATCH_SIZE = 2
    EPOCH = 'epoch'
    TIMESTAMP = "timestamp"
    METRICS_FOLDER = 'metrics'
    TRAINING_FOLDER = "training"
    TESTING_FOLDER  = 'testing'
    FILE_NAME_CONVENTION = 'file_name'