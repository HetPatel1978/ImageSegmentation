# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import transforms

# class NewspaperModel(nn.Module):
#     def __init__(self):
#         super(NewspaperModel, self).__init__()
        
#         # Encoder
#         self.encoder = nn.ModuleList([
#             # First encoder block - Note: input channels changed to 1
#             nn.Sequential(
#                 nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Changed input channels to 1
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(inplace=True)
#             ),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # Second encoder block
#             nn.Sequential(
#                 nn.Conv2d(64, 128, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(128),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(128, 128, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(128),
#                 nn.ReLU(inplace=True)
#             )
#         ])
        
#         # Decoder
#         self.decoder = nn.ModuleList([
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
#             nn.Sequential(
#                 nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Changed input channels to 64
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(inplace=True)
#             ),
#             nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
#             nn.Sequential(
#                 nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Changed to output 1 channel directly
#                 nn.BatchNorm2d(1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(1, 1, kernel_size=3, padding=1),  # Final 1x1 convolution
#                 nn.BatchNorm2d(1),
#                 nn.ReLU(inplace=True)
#             )
#         ])
    
#     def forward(self, x):
#         features = []
        
#         # Encoder path
#         for i, layer in enumerate(self.encoder):
#             x = layer(x)
#             if isinstance(layer, nn.Sequential):
#                 features.append(x)
        
#         # Decoder path with modified skip connections
#         for i, layer in enumerate(self.decoder):
#             x = layer(x)
        
#         return x

# # Modified preprocessing function to convert to grayscale
# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     # Convert to grayscale
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = cv2.resize(image, (256, 256))
#     # Normalize to [0,1] and add channel dimension
#     image = image.astype(np.float32) / 255.0
#     image = np.expand_dims(image, axis=0)  # Add channel dimension
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return torch.from_numpy(image)

# # Inference function
# def predict_mask(model, image_path):
#     image = preprocess_image(image_path)
#     model.eval()
#     with torch.no_grad():
#         output = model(image)
#         output = torch.sigmoid(output)
#         mask = output.squeeze().cpu().numpy()
#         mask = (mask > 0.5).astype(np.uint8) * 255
#     return mask

# # Visualization function
# def visualize(model, image_path):
#     original = cv2.imread(image_path)
#     original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
#     original = cv2.resize(original, (256, 256))
#     mask = predict_mask(model, image_path)
    
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(original)
#     plt.title("Original Image")
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(mask, cmap="gray")
#     plt.title("Predicted Mask")
    
#     plt.show()

# def main():
#     # Initialize model
#     model = NewspaperModel()
    
#     # Load the trained weights
#     state_dict = torch.load(r"C:/Users/Het/Desktop/Het/Method-2/newspaper_unet.pth", 
#                            map_location=torch.device("cpu"))
#     model.load_state_dict(state_dict)
#     model.eval()

#     # Test the model
#     test_image = r"C:/Users/Het/Desktop/Het/Method-2/top-10-english-newspapers-in-india1.png"
#     visualize(model, test_image)

# if __name__ == "__main__":
#     main()









# try and error 


import cv2
import pytesseract
from PIL import Image
import numpy as np
import joblib  # For loading pre-trained models (if saved)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Stage 1: OCR Extraction ---

def extract_text_from_image(image_path):
    """
    Load an image, preprocess it, and extract text using pytesseract.
    """
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    
    # Optionally, you can add pre-processing steps here (e.g., grayscale, thresholding)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # For example, applying a threshold might improve OCR results:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert thresholded image to PIL format for pytesseract
    pil_image = Image.fromarray(thresh)
    # Extract text from image
    text = pytesseract.image_to_string(pil_image)
    return text

# --- Stage 2: Text Classification ---

# Load your pre-trained TF-IDF vectorizer and classifier (if available)
# For example, if you've saved them using joblib:
# vectorizer = joblib.load('tfidf_vectorizer.pkl')
# classifier = joblib.load('text_classifier.pkl')

# For demonstration, here we create dummy objects.
# In practice, you would train these on your labeled newspaper text dataset.
vectorizer = TfidfVectorizer()
classifier = LogisticRegression()

def train_dummy_classifier():
    """
    Dummy training function.
    Replace this with your actual training on labeled data.
    """
    # Example training data and labels
    X_train = [
        "The government passed a new law today",
        "The local team won the championship",
        "Stock markets saw a significant rise",
        "New movie release breaks box office records"
    ]
    y_train = ["politics", "sports", "business", "entertainment"]
    
    # Fit the TF-IDF vectorizer and transform the training texts
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Train the classifier
    classifier.fit(X_train_tfidf, y_train)

# Uncomment the following line to train the dummy classifier.
# In your production pipeline, you would load a pre-trained model.
# train_dummy_classifier()

def classify_text(text):
    """
    Classify the extracted text into a category.
    """
    # Transform text using the vectorizer
    text_tfidf = vectorizer.transform([text])
    # Predict the category
    predicted_category = classifier.predict(text_tfidf)[0]
    return predicted_category

# --- Integration: Full Pipeline Example ---

def process_and_classify(image_path):
    """
    Full pipeline: extract text from image and classify it.
    """
    # Extract text via OCR
    extracted_text = extract_text_from_image(image_path)
    print("Extracted Text:")
    print(extracted_text)
    
    # Classify the extracted text (ensure your classifier is trained or loaded)
    try:
        category = classify_text(extracted_text)
        print(f"Predicted Category: {category}")
    except Exception as e:
        print(f"Classification failed: {e}")

# Example usage:
if __name__ == "__main__":
    test_image_path = r"C:/Users/Het/Desktop/Het/Surya/results/surya/imageprocessor/imageprocessor_0_text.png"
    process_and_classify(test_image_path)


