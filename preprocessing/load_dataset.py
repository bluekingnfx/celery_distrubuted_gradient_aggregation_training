from pathlib import Path
from os import listdir
from cv2 import imread,cvtColor, COLOR_BGR2RGB,resize
from sklearn.model_selection import train_test_split
import numpy as np


from helpers.global_config import Config

def load_data():
    data_dir = Path('vanitha_mam_breast_cancer_dataset')
    no_cancer = data_dir / str(Config.NON_CANCEROUS)
    cancerous = data_dir / str(Config.CANCEROUS)
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
    
    
    x_train,x_test,y_train,y_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True,stratify=labels)
    
    Path('preprocessed').mkdir(exist_ok=True)
    np.save('preprocessed/x_train.npy', x_train)
    np.save('preprocessed/y_train.npy', y_train)
    np.save('preprocessed/x_test.npy', x_test)
    np.save('preprocessed/y_test.npy', y_test)
    
    print("COmpleted!")
    
if __name__ == "__main__":
    load_data()


