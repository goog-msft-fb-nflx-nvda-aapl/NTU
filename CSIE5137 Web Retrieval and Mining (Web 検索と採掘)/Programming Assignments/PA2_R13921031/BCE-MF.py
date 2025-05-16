#!/usr/bin/env python3
import pandas as pd
import numpy as np
import time
import os
import sys
import pickle
import argparse
from tqdm import tqdm

class BCE_MF:
    """
    Binary Cross-Entropy Matrix Factorization for implicit feedback recommendation
    
    This model uses binary cross-entropy loss to optimize user and item embeddings
    for recommendation tasks with implicit feedback (e.g., clicks, views).
    """
    
    def __init__(self, n_users, n_items, n_factors=32, lr=0.01, reg=0.01, n_epochs=10, neg_pos_ratio=0.5):
        # Basic model parameters
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors  # dimension of latent factors
        
        # Training parameters
        self.lr = lr  # learning rate
        self.reg = reg  # regularization strength
        self.n_epochs = n_epochs
        self.neg_pos_ratio = neg_pos_ratio  # ratio of negative to positive samples
        
        # Initialize latent factors with small random values
        # Normal distribution helps prevent extreme initial predictions
        self.user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
    
    def sigmoid(self, x):
        # Sigmoid with clipping to prevent numerical overflow
        return 1.0 / (1.0 + np.exp(np.clip(-x, -30, 30)))
    
    def fit(self, user_items_dict, idx_to_item=None, output_path=None):
        """
        Train the model using BCE loss
        
        Args:
            user_items_dict: Dictionary mapping user IDs to lists of interacted item indices
            idx_to_item: Dictionary mapping internal item indices to original item IDs
            output_path: Path to save model checkpoints and recommendations
        """
        print(f"Training BCE-MF model with {self.n_factors} factors and neg_pos_ratio={self.neg_pos_ratio}...")
        
        # Get all possible items for negative sampling
        all_items = set(range(self.n_items))
        
        # Main training loop
        for epoch in range(self.n_epochs):
            start_time = time.time()
            loss = 0
            
            # Process each user's interactions
            for user_id, pos_items in tqdm(user_items_dict.items(), desc=f"Epoch {epoch+1}/{self.n_epochs}"):
                pos_items_set = set(pos_items)
                neg_items = list(all_items - pos_items_set)  # Items the user hasn't interacted with
                
                # Get user embedding vector for this update
                user_vec = self.user_factors[user_id]
                
                # Process positive samples (items the user has interacted with)
                for pos_item in pos_items:
                    pos_item_vec = self.item_factors[pos_item]
                    
                    # Compute prediction and loss for positive item
                    pos_pred = np.dot(user_vec, pos_item_vec)
                    pos_prob = self.sigmoid(pos_pred)
                    pos_loss = -np.log(pos_prob)  # BCE loss for positive example
                    loss += pos_loss
                    
                    # Gradient for positive example
                    pos_grad = (pos_prob - 1)  # d(-log(sigmoid(x)))/dx = sigmoid(x) - 1
                    
                    # Compute gradients and update parameters
                    user_grad = pos_grad * pos_item_vec + self.reg * user_vec
                    item_grad = pos_grad * user_vec + self.reg * pos_item_vec
                    
                    # SGD update
                    self.user_factors[user_id] -= self.lr * user_grad
                    self.item_factors[pos_item] -= self.lr * item_grad
                
                # Determine number of negative samples based on ratio
                n_pos = len(pos_items)
                n_neg = max(1, int(n_pos * self.neg_pos_ratio))
                
                # Sample negative items
                if n_neg <= len(neg_items):
                    neg_samples = np.random.choice(neg_items, size=n_neg, replace=False)
                else:
                    # Fall back to sampling with replacement if we need more than available
                    neg_samples = np.random.choice(neg_items, size=n_neg, replace=True)
                
                # Process negative samples
                for neg_item in neg_samples:
                    neg_item_vec = self.item_factors[neg_item]
                    
                    # Compute prediction and loss for negative item
                    neg_pred = np.dot(user_vec, neg_item_vec)
                    neg_prob = self.sigmoid(neg_pred)
                    neg_loss = -np.log(1 - neg_prob)  # BCE loss for negative example
                    loss += neg_loss
                    
                    # Gradient for negative example
                    neg_grad = neg_prob  # d(-log(1-sigmoid(x)))/dx = sigmoid(x)
                    
                    # Compute gradients and update parameters
                    user_grad = neg_grad * neg_item_vec + self.reg * user_vec
                    item_grad = neg_grad * user_vec + self.reg * neg_item_vec
                    
                    # SGD update
                    self.user_factors[user_id] -= self.lr * user_grad
                    self.item_factors[neg_item] -= self.lr * item_grad
            
            # Report progress
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1} completed in {elapsed:.2f}s, loss: {loss}")
            
            # Save model and generate recommendations at final epoch
            is_final_epoch = (epoch + 1) == self.n_epochs
            if output_path and idx_to_item is not None and is_final_epoch:
                # Save model state
                model_path = f"{output_path}_model_epoch_{epoch+1}.pkl"
                self.save_model(model_path)
                print(f"Model saved to {model_path}")
                
                # Generate and save top-k recommendations for each user
                recommendations_path = f"{output_path}_recommendations_epoch_{epoch+1}.csv"
                recommendations = {}
                for user_id in user_items_dict.keys():
                    recommendations[user_id] = self.recommend(user_id, user_items_dict, idx_to_item, top_k=50)
                save_recommendations(recommendations, recommendations_path)
                print(f"Recommendations saved to {recommendations_path}")
    
    def save_model(self, file_path):
        """Save the model parameters to a pickle file"""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'n_users': self.n_users,
                'n_items': self.n_items,
                'n_factors': self.n_factors,
                'lr': self.lr,
                'reg': self.reg,
                'n_epochs': self.n_epochs,
                'neg_pos_ratio': self.neg_pos_ratio
            }, f)
    
    @classmethod
    def load_model(cls, file_path):
        """Load a saved model from a pickle file"""
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model with saved parameters
        model = cls(
            n_users=model_data['n_users'],
            n_items=model_data['n_items'],
            n_factors=model_data['n_factors'],
            lr=model_data['lr'],
            reg=model_data['reg'],
            n_epochs=model_data['n_epochs'],
            neg_pos_ratio=model_data.get('neg_pos_ratio', 0.5)  # For compatibility
        )
        
        # Restore learned parameters
        model.user_factors = model_data['user_factors']
        model.item_factors = model_data['item_factors']
        
        return model
    
    def predict(self, user_id, item_id):
        """Calculate the preference score of a user for an item"""
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])
    
    def recommend(self, user_id, user_items_dict, idx_to_item, top_k=50):
        """
        Generate top-k item recommendations for a user
        
        Args:
            user_id: The user to generate recommendations for
            user_items_dict: Dictionary of user interaction history
            idx_to_item: Mapping from internal indices to original item IDs
            top_k: Number of recommendations to generate
            
        Returns:
            List of recommended item IDs
        """
        # Skip items the user has already interacted with
        interacted_items = set(user_items_dict.get(user_id, []))
        
        # Calculate scores for all items using dot product
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)
        
        # Filter out items the user has already interacted with
        candidate_items = np.array([i for i in range(self.n_items) if i not in interacted_items])
        candidate_scores = scores[candidate_items]
        
        # Get top-k items by score
        top_indices = candidate_scores.argsort()[::-1][:top_k]  # Sort in descending order
        top_items_idx = candidate_items[top_indices]
        
        # Map back to original item IDs
        top_items = [idx_to_item[idx] for idx in top_items_idx]
        
        return top_items


def load_data(file_path):
    """
    Load interaction data from CSV file
    
    Returns:
        user_items_dict: Dictionary mapping user IDs to lists of item indices
        users: Array of unique user IDs
        items: Array of unique item IDs
        item_to_idx: Mapping from original item IDs to internal indices
        idx_to_item: Mapping from internal indices to original item IDs
    """
    df = pd.read_csv(file_path)
    
    # Extract unique users
    users = df['UserId'].unique()
    
    # Extract unique items (handling space-separated lists)
    items = df['ItemId'].str.split().explode().astype(int).unique()
    
    # Create mappings between internal indices and original IDs
    item_to_idx = {item_id: idx for idx, item_id in enumerate(items)}
    idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
    
    # Create user-item interaction dictionary
    user_items_dict = {}
    for _, row in df.iterrows():
        user_id = row['UserId']
        items_str = row['ItemId']
        item_ids = [item_to_idx[int(i)] for i in items_str.split()]
        user_items_dict[user_id] = item_ids
    
    return user_items_dict, users, items, item_to_idx, idx_to_item

def save_recommendations(recommendations_dict, output_path):
    """Save recommendations to CSV file"""
    with open(output_path, 'w') as f:
        f.write("UserId,ItemId\n")
        for user_id, items in recommendations_dict.items():
            items_str = ' '.join(map(str, items))
            f.write(f"{user_id},{items_str}\n")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BCE-MF Recommendation System')
    
    # Required argument
    parser.add_argument('output_path', type=str, help='Base path for saving results')
    
    # Optional arguments with defaults
    parser.add_argument('--n_factors', type=int, default=64, help='Number of latent factors')
    parser.add_argument('--neg_pos_ratio', type=float, default=0.5, help='Negative to positive sampling ratio')
    parser.add_argument('--n_epochs', type=int, default=32, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--reg', type=float, default=0.01, help='Regularization strength')
    parser.add_argument('--train_path', type=str, default='train.csv', help='Path to training data')
    
    return parser.parse_args()

def main():
    # Get command line arguments
    args = parse_args()
    
    # Create output path with model hyperparameters
    output_path = f"{args.output_path}_factors{args.n_factors}_npr{args.neg_pos_ratio}"
    print(f"Results will be saved to: {output_path}")
    
    # Load training data
    print("Loading data...")
    user_items_dict, users, items, item_to_idx, idx_to_item = load_data(args.train_path)
    n_users = len(users)
    n_items = len(items)
    
    print(f"Loaded {n_users} users and {n_items} unique items")
    print(f"Original item ID range: {min(items)} to {max(items)}")
    
    # Create and train the model
    model = BCE_MF(
        n_users=n_users,
        n_items=n_items,
        n_factors=args.n_factors,
        lr=args.lr,
        reg=args.reg,
        n_epochs=args.n_epochs,
        neg_pos_ratio=args.neg_pos_ratio
    )
    
    # Train the model
    model.fit(user_items_dict, idx_to_item, output_path)
    
    print(f"Training completed. Results saved to {output_path}_recommendations_epoch_{args.n_epochs}.csv")

if __name__ == "__main__":
    main()