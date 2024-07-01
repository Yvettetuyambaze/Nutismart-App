from src.data_preprocessing import preprocess_data
from src.model_training import create_model, train_model
from src.calorie_estimation import estimate_portion_size, estimate_calories
from src.recommender import FoodRecommender
import tensorflow as tf
import pandas as pd
# Load the CSV file
food_data = pd.read_csv('data/nutrition/rwandan_food_data.csv')

import os

def main():
    # Data preprocessing
    data_dir = './Dataset/Rwandan dishes'
    train_generator, validation_generator = preprocess_data(data_dir)
    
    # Model training
    num_classes = 16  # Number of Rwandan dishes in your dataset
    model = create_model(num_classes)
    history = train_model(model, train_generator, validation_generator)
    model.save('./models/rwandan_food_model.h5')
    
    # Load trained model for prediction
    loaded_model = tf.keras.models.load_model('./models/rwandan_food_model.h5')
    
    # Example prediction (in a real scenario, this would come from the model)
    food_name = 'Ugali'  # Example dish
    
    # Get a sample image path
    sample_image_path = os.path.join(data_dir, food_name, os.listdir(os.path.join(data_dir, food_name))[0])
    
    # Calorie estimation
    portion_size = estimate_portion_size(sample_image_path, reference_object_size=1)
    calories = estimate_calories(food_name, portion_size)
    print(f"Estimated calories for {food_name}: {calories}")
    
    # Recommendation
    food_data = pd.read_csv('./data/nutrition/rwandan_food_data.csv')
    recommender = FoodRecommender(food_data)
    recommendations = recommender.get_recommendations(food_name)
    print(f"Recommendations for {food_name}: {recommendations}")

if __name__ == "__main__":
    main()