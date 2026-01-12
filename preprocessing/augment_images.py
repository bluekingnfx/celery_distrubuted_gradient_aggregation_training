
import albumentations as A
from tensorflow import Tensor

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
    A.Normalize()
])
    
def augment_images(img, label, transformation):
    if hasattr(img,'numpy'):
        img = img.numpy()
    if hasattr(img, 'numpy'):
        label = label.numpy()
    
    print(f'original img_shape: {img.shape} \n {img.dtype}')
    
    augmented_image = transformation(image=img)['image']
    
    print(f'Augmented img_shape: {augmented_image.shape} \n {augmented_image.dtype}')
    
    return augmented_image, label
    
    
    
    
    
    