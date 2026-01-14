
import albumentations as A
from tensorflow import is_tensor, convert_to_tensor, float32 as TfFloat32
import numpy as np

from helpers.global_config import Config

augment_pipeline_train = A.Compose([
    A.Resize(Config.INPUT_SHAPE[0], Config.INPUT_SHAPE[1]),
    A.RandomRotate90(p = 0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3,7), p = 0.3),
        A.MotionBlur(blur_limit=(3,7), p = 0.3)
    ]),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CLAHE(clip_limit= 2, p = 0.3)
    ]),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0)
])


def augment_images(img, label, transformation):
    if is_tensor(img):
        img = img.numpy()
        img = img*255 if img.max() <=1 else img
        img = img.astype(np.uint8) 
    
    if is_tensor(label):
        label = label.numpy()
        label = label.astype(np.float32)
        label = convert_to_tensor(label,TfFloat32)
    
    augmented_image = transformation(image=img)['image']
    
    augmented_image = convert_to_tensor(augmented_image,dtype=TfFloat32)
    
    return augmented_image, label



augment_pipeline_test = A.Compose([
    A.Resize(height=Config.INPUT_SHAPE[0], width=Config.INPUT_SHAPE[1]),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0)
])



def test_augment_image(img, label, transform):
    
    if is_tensor(img):
        img = img.numpy()
        img = img*255 if img.max() <=1 else img
        img = img.astype(np.uint8)
    
    if is_tensor(label):
        label = label.numpy()
        label = label.astype(np.float32)
        label = convert_to_tensor(label, dtype=TfFloat32)
    
    augmented_img = transform(image=img)['image']
    augmented_img = convert_to_tensor(augmented_img, TfFloat32)
    
    return augmented_img, label