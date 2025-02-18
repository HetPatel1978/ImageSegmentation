import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def verify_model(model_path):
    print("Loading and verifying model...")
    model = load_model(model_path, compile=False)
    
    print("\nModel Architecture:")
    model.summary()
    
    print("\nOutput Layer Details:")
    output_layer = model.layers[-1]
    print(f"Output layer type: {type(output_layer).__name__}")
    print(f"Output layer activation: {output_layer.activation.__name__}")
    print(f"Output layer units/filters: {output_layer.filters}")
    
    # Check if model architecture matches segmentation requirements
    expected_classes = 7  # background + 6 classes
    actual_classes = output_layer.filters
    
    if actual_classes != expected_classes:
        print(f"\nWARNING: Model output channels ({actual_classes}) does not match expected number of classes ({expected_classes})")
        print("The model needs to be retrained with the correct number of output classes")
        
    return model

if __name__ == "__main__":
    model_path = r"C:/Users/Het/Desktop/Het/unet_model.h5"
    model = verify_model(model_path)