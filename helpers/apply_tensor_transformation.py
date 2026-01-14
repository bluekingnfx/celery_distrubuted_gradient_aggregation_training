from tensorflow import data,py_function, float32 as TFloat32
from numpy import array

from set_shapes import set_shapes
from preprocessing.augment_images import augment_pipeline_train, augment_images, augment_pipeline_test, test_augment_image

def applyTensorTransformation(
    X, 
    Y,
    shuffle_no = 200,
    batch_no = 2,
    prefetch_no = 1,
    train_preset = True
):
    dataset = data.Dataset.from_tensor_slices((X,Y))
    dataset = dataset.map(
        lambda x, y: py_function(
            func = lambda img, label: augment_images(img, label, augment_pipeline_train) if train_preset else test_augment_image(img,label,augment_pipeline_test),
            inp = [x,y],
            Tout = [TFloat32,TFloat32]
        )
    )
    
    dataset = dataset.map(set_shapes).shuffle(shuffle_no).batch(batch_no).prefetch(prefetch_no)
    
    return dataset


def applyTensorTransformationForX_VAL(
    X_Val, 
    Y_Val,
    shuffle_no = 20,
    batch_no = 2,
    prefetch_no = 1,
):
    dataset = applyTensorTransformation(
        X_Val, Y_Val, shuffle_no=shuffle_no, batch_no=batch_no, prefetch_no=prefetch_no, train_preset=False)
    X_Val = dataset.map(lambda x, _: x)
    Y_Val = dataset.map(lambda _, y: y).as_numpy_iterator()
    Y_Val = array([y for y in Y_Val])
    
    return X_Val, Y_Val
    