import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def create_model(num_classes):
    """
    Create and compile the model for training.
    
    Args:
    num_classes (int): Number of classes (Rwandan dishes) to classify.
    
    Returns:
    tf.keras.Model: Compiled model ready for training.
    """
    # Load pre-trained InceptionV3 model without top layers
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, train_generator, validation_generator, epochs=50):
    """
    Train the model using the provided data generators.
    
    Args:
    model (tf.keras.Model): The model to train.
    train_generator (tf.keras.preprocessing.image.DirectoryIterator): Training data generator.
    validation_generator (tf.keras.preprocessing.image.DirectoryIterator): Validation data generator.
    epochs (int): Number of epochs to train for.
    
    Returns:
    tf.keras.callbacks.History: Training history.
    """
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
        
    )
    
    return history
