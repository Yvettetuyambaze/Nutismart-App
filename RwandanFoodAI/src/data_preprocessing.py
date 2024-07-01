import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(data_dir, target_size=(299, 299), batch_size=32):
    """
    Preprocess and augment image data for model training.
    
    Args:
    data_dir (str): Directory containing the image dataset.
    target_size (tuple): Size to which all images will be resized.
    batch_size (int): Number of images per batch.
    
    Returns:
    tuple: Training and validation data generators.
    """
    # Create an ImageDataGenerator with data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Create a generator for training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Create a generator for validation data
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator