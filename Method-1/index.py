# import os
# import cv2
# import numpy as np
# import logging
# import xml.etree.ElementTree as ET
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Paths to datasets
# images_path = r'C:/Users/Het/Desktop/Het/PRImA Layout Analysis Dataset/Images'
# xml_path = r'C:/Users/Het/Desktop/Het/PRImA Layout Analysis Dataset/XML'

# # Image dimensions
# IMG_HEIGHT, IMG_WIDTH = 256, 256

# # Function to parse XML annotation and create masks
# def parse_xml_annotation(xml_file, img_shape):
#     mask = np.zeros(img_shape[:2], dtype=np.uint8)
#     try:
#         tree = ET.parse(xml_file)
#         root = tree.getroot()
#         for region in root.findall(".//Region[@type='TextRegion']"):
#             points = region.find("Coords").attrib['points']
#             pts = np.array([[int(coord.split(",")[0]), int(coord.split(",")[1])] for coord in points.split()], dtype=np.int32)
#             cv2.fillPoly(mask, [pts], 255)
#     except Exception as e:
#         logging.error(f"Error parsing {xml_file}: {e}")
#     return mask

# # Function to load dataset
# def load_dataset(images_path, xml_path):
#     images = []
#     masks = []

#     image_files = set(f for f in os.listdir(images_path) if f.endswith('.tif'))
#     xml_files = set(f for f in os.listdir(xml_path) if f.endswith('.xml'))

#     missing_xmls = {f.replace('.tif', '.xml') for f in image_files} - xml_files
#     if missing_xmls:
#         logging.warning(f"Missing XML files for images: {missing_xmls}")

#     for img_file in image_files:
#         xml_file = os.path.join(xml_path, img_file.replace('.tif', '.xml'))
#         img_path = os.path.join(images_path, img_file)

#         if not os.path.exists(xml_file):
#             logging.warning(f"Skipping {img_file} - XML file missing")
#             continue

#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

#         mask = parse_xml_annotation(xml_file, image.shape)
#         mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))

#         images.append(image)
#         masks.append(mask)

#     return np.array(images), np.array(masks)

# # U-Net model definition
# def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
#     inputs = Input(input_size)

#     # Encoder
#     conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     # Bottleneck
#     conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

#     # Decoder
#     up1 = UpSampling2D(size=(2, 2))(conv3)
#     merge1 = concatenate([conv2, up1], axis=3)
#     conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge1)
#     conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

#     up2 = UpSampling2D(size=(2, 2))(conv4)
#     merge2 = concatenate([conv1, up2], axis=3)
#     conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge2)
#     conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # Main code
# if __name__ == "__main__":
#     logging.info("Loading dataset...")
#     images, masks = load_dataset(images_path, xml_path)
#     masks = np.expand_dims(masks, axis=-1)  # Add channel dimension

#     logging.info(f"Dataset loaded: {len(images)} images")

#     # Split dataset
#     X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

#     # Initialize and train the model
#     model = unet_model()
#     model.summary()

#     logging.info("Starting training...")
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         batch_size=8,
#         epochs=20
#     )

#     # Save the model
#     model.save("unet_model.h5")
#     logging.info("Model saved as unet_model.h5")

#     # Visualize training history
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.legend()
#     plt.show()

#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.legend()
#     plt.show()













import os
import cv2
import numpy as np
import logging
import xml.etree.ElementTree as ET
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to datasets
images_path = r'C:/Users/Het/Desktop/Het/Method-1/PRImA Layout Analysis Dataset/Images'
xml_path = r'C:/Users/Het/Desktop/Het/Method-1/PRImA Layout Analysis Dataset/XML'

# Image dimensions and number of classes
IMG_HEIGHT, IMG_WIDTH = 256, 256
NUM_CLASSES = 4  # Background, Text, Image, Table

# Function to parse XML annotation and create masks
def parse_xml_annotation(xml_file, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for region in root.findall(".//Region"):
            region_type = region.attrib.get("type", "Unknown")

            if region_type == "TextRegion":
                label = 1
            elif region_type == "ImageRegion":
                label = 2
            elif region_type == "TableRegion":
                label = 3
            else:
                label = 0  # Background

            points = region.find("Coords").attrib['points']
            pts = np.array([[int(coord.split(",")[0]), int(coord.split(",")[1])] for coord in points.split()], dtype=np.int32)
            cv2.fillPoly(mask, [pts], label)
    except Exception as e:
        logging.error(f"Error parsing {xml_file}: {e}")
    return mask

# Function to load dataset
def load_dataset(images_path, xml_path):
    images = []
    masks = []

    image_files = set(f for f in os.listdir(images_path) if f.endswith('.tif'))
    xml_files = set(f for f in os.listdir(xml_path) if f.endswith('.xml'))

    for img_file in image_files:
        xml_file = os.path.join(xml_path, img_file.replace('.tif', '.xml'))
        img_path = os.path.join(images_path, img_file)

        if not os.path.exists(xml_file):
            logging.warning(f"Skipping {img_file} - XML file missing")
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

        mask = parse_xml_annotation(xml_file, image.shape)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        mask = to_categorical(mask, num_classes=NUM_CLASSES)  # One-hot encode mask

        images.append(image)
        masks.append(mask)

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

# U-Net model definition
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv3)
    merge1 = concatenate([conv2, up1], axis=3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    merge2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge2)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    # Multi-class output layer
    outputs = Conv2D(num_classes, (1, 1), activation="softmax")(conv5)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Main code
if __name__ == "__main__":
    logging.info("Loading dataset...")
    images, masks = load_dataset(images_path, xml_path)
    logging.info(f"Dataset loaded: {len(images)} images")

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = unet_model()
    model.summary()

    logging.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=8,
        epochs=20
    )

    # Save the model
    model.save("unet_multi_class.h5")
    logging.info("Model saved as unet_multi_class.h5")

    # Visualize training history
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()
