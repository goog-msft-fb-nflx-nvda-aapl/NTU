#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys
import pickle
from tqdm import tqdm

class BPR_MF:
    """
    Bayesian Personalized Ranking Matrix Factorization for Recommendation Systems
    
    This implementation follows the BPR optimization criterion using stochastic gradient
    descent to learn the latent factors for users and items.
    
    Reference: Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009).
    BPR: Bayesian personalized ranking from implicit feedback.
    """
    
    def __init__(self, n_users, n_items, n_factors=32, lr=0.01, reg=0.01, n_epochs=10, neg_pos_ratio=0.5):
        """
        Initialize the BPR-MF model
        
        Args:
            n_users: Total number of users in the dataset
            n_items: Total number of items in the dataset
            n_factors: Number of latent factors for matrix factorization
            lr: Learning rate for gradient descent
            reg: Regularization strength to prevent overfitting
            n_epochs: Number of training epochs
            neg_pos_ratio: Sampling ratio of negative to positive examples
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.neg_pos_ratio = neg_pos_ratio
        
        # Initialize the latent factors with small random values to break symmetry
        self.user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
    
    @classmethod
    def load_model(cls, file_path):
        """
        Load a previously saved model from disk
        
        Args:
            file_path: Path to the pickle file containing the model
            
        Returns:
            An instance of BPR_MF with loaded parameters
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create a fresh instance with the saved parameters
            model = cls(
                n_users=model_data['n_users'],
                n_items=model_data['n_items'],
                n_factors=model_data['n_factors'],
                lr=model_data['lr'],
                reg=model_data['reg'],
                n_epochs=model_data['n_epochs'],
                # Handle both new and old model formats
                neg_pos_ratio=model_data.get('neg_pos_ratio', model_data.get('pos_neg_ratio', 0.5))
            )
            
            # Load the learned factors
            model.user_factors = model_data['user_factors']
            model.item_factors = model_data['item_factors']
            
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def predict(self, user_id, item_id):
        """
        Calculate preference score for a user-item pair
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            
        Returns:
            Predicted preference score
        """
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])
    
    def recommend(self, user_id, user_items_dict, top_k=50):
        """
        Generate top-k recommendations for a specific user
        
        Args:
            user_id: ID of the target user
            user_items_dict: Dictionary mapping users to their interacted items
            top_k: Number of recommendations to generate
            
        Returns:
            List of recommended item IDs
        """
        # Get the items this user has already interacted with
        interacted_items = set(user_items_dict.get(user_id, []))
        
        # Compute scores for all items at once (vectorized operation)
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)
        
        # Filter out items the user has already interacted with
        candidate_items = np.array([i for i in range(self.n_items) if i not in interacted_items])
        
        if len(candidate_items) == 0:
            return []  # No candidates available
            
        # Extract scores for candidate items only
        candidate_scores = scores[candidate_items]
        
        # Get top-k items with highest scores
        top_indices = candidate_scores.argsort()[::-1][:top_k]
        top_items = candidate_items[top_indices]
        
        return top_items


def load_data(file_path):
    """
    Load and process user-item interaction data
    
    Args:
        file_path: Path to the CSV file containing user-item interactions
        
    Returns:
        Dictionary mapping user IDs to lists of interacted item IDs
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert CSV data to user-item dictionary
        user_items_dict = {}
        for _, row in df.iterrows():
            user_id = row['UserId']
            items_str = row['ItemId']
            item_ids = [int(i) for i in items_str.split()]
            user_items_dict[user_id] = item_ids
        
        return user_items_dict
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def save_recommendations(recommendations_dict, output_path):
    """
    Save generated recommendations to a CSV file
    
    Args:
        recommendations_dict: Dictionary mapping user IDs to recommended item IDs
        output_path: Path where the recommendations will be saved
    """
    try:
        with open(output_path, 'w') as f:
            f.write("UserId,ItemId\n")  # CSV header
            for user_id, items in recommendations_dict.items():
                items_str = ' '.join(map(str, items))
                f.write(f"{user_id},{items_str}\n")
        print(f"Successfully saved recommendations to {output_path}")
    except Exception as e:
        print(f"Error saving recommendations: {e}")
        sys.exit(1)


def main():
    """Main function to load data, model and generate recommendations"""
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python BPR-MF.py <output_path>")
        sys.exit(1)
    
    output_path = sys.argv[1]
    
    # File paths
    train_path = 'train.csv'
    model_path = 'best_model.pkl'
    
    # Data loading
    print("Loading training data...")
    user_items_dict = load_data(train_path)
    
    # Model loading
    print(f"Loading pre-trained BPR-MF model from {model_path}...")
    model = BPR_MF.load_model(model_path)
    
    print(f"Model loaded successfully with {model.n_users} users, {model.n_items} items, and {model.n_factors} latent factors")
    
    # Generate recommendations for all users
    print("Generating personalized recommendations...")
    recommendations = {}
    for user_id in tqdm(user_items_dict.keys(), desc="Users processed"):
        recommendations[user_id] = model.recommend(user_id, user_items_dict, top_k=50)
    
    # Save results
    print(f"Saving recommendation results to {output_path}...")
    save_recommendations(recommendations, output_path)
    print("Recommendation process completed!")


if __name__ == "__main__":
    main()
