import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
import logging
from datetime import datetime

def build_unet(input_size=(256, 256, 3), num_classes=7):
    """
    Build U-Net architecture for multi-class layout segmentation
    
    Args:
        input_size: Input image dimensions (height, width, channels)
        num_classes: Number of segmentation classes (including background)
    """
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    # Output layer - Changed to support multi-class segmentation
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class PrimaLayoutAnalysis:
    def __init__(self, images_path, xml_path, target_size=(256, 256), log_file=None):
        self.images_path = images_path
        self.xml_path = xml_path
        self.target_size = target_size
        
        # Define class mappings
        self.classes = {
            'background': 0,  # Added background class explicitly
            'text': 1,
            'image': 2,
            'table': 3,
            'math': 4,
            'separator': 5,
            'other': 6
        }
        
        # Set up logging
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'layout_analysis_{timestamp}.log'
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def train(self, batch_size=8, epochs=50, validation_split=0.2):
        """Train the layout analysis model"""
        try:
            # Load dataset
            logging.info("Loading dataset...")
            images, masks = self.load_dataset()
            
            # Build model with correct number of classes
            logging.info("Building U-Net model...")
            input_size = (*self.target_size, 3)
            model = build_unet(input_size=input_size, num_classes=len(self.classes))
            
            logging.info("Model Summary:")
            model.summary(print_fn=logging.info)
            
            # Train model
            logging.info("Starting training...")
            history = model.fit(
                images, masks,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                verbose=1
            )
            
            return model, history
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

def main():
    # Configuration
    images_path = r"C:/Users/Het/Desktop/Het/PRImA Layout Analysis Dataset/Images"
    xml_path = r"C:/Users/Het/Desktop/Het/PRImA Layout Analysis Dataset/XML"
    output_model_path = "prima_layout_model_multiclass.h5"
    
    try:
        # Initialize analyzer
        analyzer = PrimaLayoutAnalysis(images_path, xml_path)
        
        # Train model
        logging.info("Starting training process")
        model, history = analyzer.train(epochs=50)  # Adjust epochs as needed
        
        # Save model
        model.save(output_model_path)
        logging.info(f"Model saved successfully to {output_model_path}")
        
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()