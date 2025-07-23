import cv2

def preprocess_image(image_path):
    """
    Loads the image, converts to grayscale and resize the image to fixed width whilst
    maintaining aspect ratio for better OCR output.
    """
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to a fixed width (e.g., 1000px) to improve text clarity
    target_width = 1000
    scale_ratio = target_width / gray.shape[1]
    resized = cv2.resize(gray, (target_width, int(gray.shape[0] * scale_ratio)), interpolation=cv2.INTER_LINEAR)

    # Save preprocessed image to temp file
    temp_path = "preprocessed_temp.png"
    cv2.imwrite(temp_path, resized)
    return temp_path
