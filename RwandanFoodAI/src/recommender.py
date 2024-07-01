import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class FoodRecommender:
    def __init__(self, food_data):
        """
        Initialize the FoodRecommender with food data.
        
        Args:
        food_data (pd.DataFrame): DataFrame containing food items and their nutritional information.
        """
        self.food_data = food_data
        self.similarity_matrix = self._compute_similarity()
    
    def _compute_similarity(self):
        """
        Compute the similarity matrix between food items.
        
        Returns:
        np.array: Similarity matrix.
        """
        return cosine_similarity(self.food_data.drop('name', axis=1))
    
    def get_recommendations(self, food_name, top_n=5):
        """
        Get food recommendations based on similarity to the given food.
        
        Args:
        food_name (str): Name of the food item to base recommendations on.
        top_n (int): Number of recommendations to return.
        
        Returns:
        list: List of recommended food names.
        """
        food_index = self.food_data[self.food_data['name'] == food_name].index[0]
        similar_foods = list(enumerate(self.similarity_matrix[food_index]))
        similar_foods = sorted(similar_foods, key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for i in range(1, top_n + 1):
            recommendations.append(self.food_data.iloc[similar_foods[i][0]]['name'])
        
        return recommendations