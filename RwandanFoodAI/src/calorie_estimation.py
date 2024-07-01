import cv2
import numpy as np

def estimate_portion_size(image_path, reference_object_size):
    """
    Estimate the portion size of food in an image.
    
    Args:
    image_path (str): Path to the food image.
    reference_object_size (float): Size of the reference object in square inches.
    
    Returns:
    float: Estimated size of the food portion in square inches.
    """
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is the food item
    food_contour = max(contours, key=cv2.contourArea)
    food_area = cv2.contourArea(food_contour)
    
    # Calculate the size of the food in inches
    pixels_per_inch = np.sqrt(food_area / reference_object_size)
    food_size_inches = food_area / (pixels_per_inch ** 2)
    
    return food_size_inches

def estimate_calories(food_name, portion_size):
    """
    Estimate calories for a given food and portion size.
    
    Args:
    food_name (str): Name of the food item.
    portion_size (float): Size of the portion in square inches.
    
    Returns:
    float: Estimated calories for the given food and portion size.
    """
    # This is a simplified calorie lookup table. In a real system, you'd use a comprehensive database.
    calories_per_inch = {
        'ugali': 100,
        'isombe': 50,
        'matoke': 80,
        # Add more Rwandan dishes here
    }
    
    return calories_per_inch.get(food_name, 0) * portion_size