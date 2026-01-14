from pathlib import Path
from os import listdir
from cv2 import imread,cvtColor, COLOR_BGR2RGB,resize
from sklearn.model_selection import train_test_split
import numpy as np
import sys


sys.path.append(str(Path(__file__).resolve().parent.parent))

from helpers.global_config import Config

def load_data():
    parent_dir = Path(__file__).resolve().parent.parent
    data_dir = parent_dir / "vanitha_mam_breast_cancer_dataset"
    no_cancer = data_dir / str(Config.NON_CANCEROUS)
    cancerous = data_dir / str(Config.CANCEROUS)
    
    print('Directories exists, now preprocessing images...')
    images  = []
    labels = []
    
    if no_cancer.exists():
        file_list = sorted(listdir(no_cancer))
        for file_name in file_list:
            img = imread(str(no_cancer / file_name))
            if img is not None:
                img = cvtColor(img,COLOR_BGR2RGB)
                img = resize(img, (128,128))
                images.append(img)
                labels.append(Config.NON_CANCEROUS)
    
    if cancerous.exists():
        file_list = sorted(listdir(cancerous))
        for file_name in file_list:
            img = imread(str(cancerous / file_name))
            if img is not None:
                img = cvtColor(img, COLOR_BGR2RGB)
                img = resize(img, (128,128))
                images.append(img)
                labels.append(Config.CANCEROUS)
    
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    print(f"Total images loaded: {len(images)}")
    print(f"Image data shape: {images.shape}")
    print(f"Unique labels and their counts: {np.unique(labels, return_counts=True)}")
    
    x_train,x_test,y_train,y_test = train_test_split(images, labels, test_size=0.3, random_state=42, shuffle=True,stratify=labels)
    
    x_train,x_val,y_train,y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42, shuffle=True, stratify=y_train)
    
    
    # If training speed is minimum, then we do pre-processing at beginning 
    # and save to the disk, the downside is we cannot able to resize with 
    # with different resolution in global config without running load dataset
    # again. So, I decided to accept the runtime overhead for preprocessing
    # at the beginning of each training batch of every epoch.
    """ x_val = (x_val / 255.0).astype(np.float32)
    
    x_val_list = np.array([ resize(img, (Config.INPUT_SHAPE[0], Config.INPUT_SHAPE[1])) for img in x_val ])
    
    y_val = y_val.astype(np.float32) """
    
    preprocessed_dir = parent_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    np.save(preprocessed_dir / 'x_train.npy', x_train)
    np.save(preprocessed_dir / 'y_train.npy', y_train)
    np.save(preprocessed_dir / 'x_test.npy', x_test)
    np.save(preprocessed_dir / 'y_test.npy', y_test)
    np.save(preprocessed_dir / 'x_val.npy', x_val)
    np.save(preprocessed_dir / 'y_val.npy', y_val)
    
    print("COmpleted!")
    
if __name__ == "__main__":
    load_data()