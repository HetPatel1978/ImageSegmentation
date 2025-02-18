# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Load the image
# image = cv2.imread(r'C:/Users/Het/Desktop/Het/Method-2/Screenshot 2025-02-05 094752.png')

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply GaussianBlur to reduce noise
# blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# # Use Otsu's thresholding to separate text from background
# _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Invert the threshold image to highlight text
# inverted_threshold_image = cv2.bitwise_not(threshold_image)

# # Use morphological operations to clean up the image (remove noise)
# kernel = np.ones((3, 3), np.uint8)
# cleaned_image = cv2.morphologyEx(inverted_threshold_image, cv2.MORPH_CLOSE, kernel)

# # Find contours to separate text from images
# contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Create a mask for separating text and image regions
# mask_text = np.zeros_like(gray_image)
# mask_image = np.zeros_like(gray_image)

# # Loop through contours to separate text and images
# for contour in contours:
#     area = cv2.contourArea(contour)
#     if area > 500:  # This threshold can be adjusted based on the size of the text
#         cv2.drawContours(mask_text, [contour], -1, 255, -1)  # Text regions
#     else:
#         cv2.drawContours(mask_image, [contour], -1, 255, -1)  # Image regions

# # Extract text and image regions using the masks
# text_region = cv2.bitwise_and(image, image, mask=mask_text)
# image_region = cv2.bitwise_and(image, image, mask=mask_image)

# # Display the results
# plt.figure(figsize=(10, 6))

# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
# plt.title("Grayscale Image")
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(cv2.cvtColor(text_region, cv2.COLOR_BGR2RGB))
# plt.title("Text Region")
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.imshow(cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB))
# plt.title("Image Region")
# plt.axis('off')

# plt.show()





#Try-2


# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread(r'C:/Users/Het/Desktop/Het/Method-2/top-10-english-newspapers-in-india1.png')

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply a simple threshold to separate background from foreground
# _, threshold_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)

# # Create a 4-channel image (RGBA) to add transparency
# rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

# # Set the alpha channel to 0 (transparent) where the threshold is white (background)
# rgba_image[:, :, 3] = threshold_image

# # Save or show the image with transparency
# cv2.imwrite('image_with_transparency.png', rgba_image)

# # Display the result (optional)
# cv2.imshow("Transparent Background", rgba_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()







#try-3(Working For image removel)

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def remove_background(image_path):
#     """Removes background from a newspaper image and returns the text-only mask."""
    
#     # Step 1: Load Image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not load image: {image_path}")

#     # Convert to Grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Step 2: Edge Detection (Canny)
#     edges = cv2.Canny(gray, 50, 150)

#     # Step 3: Morphological Processing
#     kernel = np.ones((3, 3), np.uint8)
#     edges_dilated = cv2.dilate(edges, kernel, iterations=2)

#     # Step 4: Otsu's Thresholding for Foreground Mask
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Step 5: GrabCut Algorithm for Fine Background Removal
#     mask = np.zeros(image.shape[:2], np.uint8)  # Initial mask
#     bgd_model = np.zeros((1, 65), np.float64)  # Background model
#     fgd_model = np.zeros((1, 65), np.float64)  # Foreground model

#     rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)  # Define a rectangle around the text
#     cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

#     # Convert GrabCut results to binary mask
#     mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

#     # Step 6: Apply Mask to Remove Background
#     no_bg = image * mask[:, :, np.newaxis]

#     return gray, edges_dilated, mask * 255, no_bg

# # ------------------- Test Background Removal ------------------- #
# def visualize_background_removal(image_path):
#     """Displays the background removal process."""
#     gray, edges, mask, result = remove_background(image_path)

#     plt.figure(figsize=(10, 5))

#     plt.subplot(1, 4, 1)
#     plt.imshow(gray, cmap="gray")
#     plt.title("Grayscale")

#     plt.subplot(1, 4, 2)
#     plt.imshow(edges, cmap="gray")
#     plt.title("Edges")

#     plt.subplot(1, 4, 3)
#     plt.imshow(mask, cmap="gray")
#     plt.title("Text Mask")

#     plt.subplot(1, 4, 4)
#     plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
#     plt.title("Background Removed")

#     plt.show()

# # ------------------- Run the Test ------------------- #
# image_path = r"C:/Users/Het/Desktop/Het/imageprocessor.jpg"
# visualize_background_removal(image_path)






#code for  saving removed photos:

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def remove_background(image_path, save_path_text, save_path_bg):
#     """Removes background from a newspaper image and saves text-only and removed background images."""
    
#     # Step 1: Load Image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not load image: {image_path}")

#     # Convert to Grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Step 2: Otsu's Thresholding for Foreground Mask
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Step 3: GrabCut Algorithm for Fine Background Removal
#     mask = np.zeros(image.shape[:2], np.uint8)
#     bgd_model = np.zeros((1, 65), np.float64)
#     fgd_model = np.zeros((1, 65), np.float64)

#     rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)
#     cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

#     # Convert GrabCut results to binary mask
#     mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

#     # Step 4: Extract Foreground (Text Only)
#     text_only = image * mask[:, :, np.newaxis]

#     # Step 5: Extract Background (Removed Portion)
#     background = image * (1 - mask[:, :, np.newaxis])

#     # Step 6: Save Results
#     cv2.imwrite(save_path_text, text_only)
#     cv2.imwrite(save_path_bg, background)

#     return text_only, background

# # ------------------- Run the Code ------------------- #
# image_path = r"C:/Users/Het/Desktop/Het/imageprocessor.jpg"
# save_text_only = r"C:/Users/Het/Desktop/Het/text_only.jpg"
# save_background = r"C:/Users/Het/Desktop/Het/removed_background.jpg"

# text_img, bg_img = remove_background(image_path, save_text_only, save_background)

# print(f"Text-only image saved at: {save_text_only}")
# print(f"Removed background image saved at: {save_background}")







