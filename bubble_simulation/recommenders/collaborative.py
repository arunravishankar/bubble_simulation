import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Set
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from ..content.item import Item
from ..users.user import User
from .base import BaseRecommender
from ..settings import (
    RECO_TOP_K, RECO_REGULARIZATION, 
    CF_NUM_FACTORS, CF_RETRAIN_FREQUENCY
)

logger = logging.getLogger(__name__)

class CollaborativeFilteringRecommender(BaseRecommender):
    """
    Matrix factorization based collaborative filtering recommender system.
    
    This recommender implements a standard matrix factorization approach using 
    singular value decomposition (SVD) to find latent factors that explain the
    user-item interaction patterns.
    """
    
    def __init__(self, 
                 name: str = "CollaborativeFilteringRecommender",
                 retrain_frequency: int = CF_RETRAIN_FREQUENCY,
                 top_k: int = RECO_TOP_K,
                 num_factors: int = CF_NUM_FACTORS,
                 regularization: float = RECO_REGULARIZATION,
                 use_implicit: bool = True):
        """
        Initialize the collaborative filtering recommender.
        
        Args:
            name: Identifier for this recommender instance
            retrain_frequency: How often to retrain the model (in timesteps)
            top_k: Default number of items to recommend
            num_factors: Number of latent factors to use
            regularization: Regularization factor to prevent overfitting
            use_implicit: Whether to use implicit feedback (ignoring rating values)
        """
        super().__init__(name=name, retrain_frequency=retrain_frequency, top_k=top_k)
        self.num_factors = num_factors
        self.regularization = regularization
        self.use_implicit = use_implicit
        
        # Model parameters to be learned
        self.user_factors = None  # User vectors in latent space
        self.item_factors = None  # Item vectors in latent space
        self.user_bias = None  # User bias terms
        self.item_bias = None  # Item bias terms
        self.global_bias = 0.0  # Global bias term
        
        # Mappings for user/item IDs to matrix indices
        self.user_id_map = {}  # User ID to matrix index
        self.item_id_map = {}  # Item ID to matrix index
        self.reverse_user_map = {}  # Matrix index to user ID
        self.reverse_item_map = {}  # Matrix index to item ID
        
    def train(self, interactions: List[Dict], items: List[Item], users: List[User]) -> None:
        """
        Train the collaborative filtering model using matrix factorization.
        
        This implementation uses Scikit-learn's TruncatedSVD for simplicity, but
        in a production system, specialized libraries like Implicit or LightFM would
        be more appropriate for implicit feedback scenarios.
        
        Args:
            interactions: List of user-item interaction dictionaries
            items: List of available content items
            users: List of users
        """
        start_time = time.time()
        
        # Create user and item ID mappings
        self.user_id_map = {user.user_id: i for i, user in enumerate(users)}
        self.item_id_map = {item.item_id: i for i, item in enumerate(items)}
        self.reverse_user_map = {i: user_id for user_id, i in self.user_id_map.items()}
        self.reverse_item_map = {i: item_id for item_id, i in self.item_id_map.items()}
        
        # Dimensions of the interaction matrix
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        
        # If no users or items, skip training
        if n_users == 0 or n_items == 0:
            logger.warning(f"{self.name}: Cannot train with 0 users or items")
            return
            
        # Convert interactions to a sparse matrix
        # Building matrix in COO format first for efficient construction
        user_indices = []
        item_indices = []
        values = []
        
        for interaction in interactions:
            user_id = interaction['user_id']
            item_id = interaction['item_id']
            engaged = interaction['engaged']
            
            # Skip if user or item not in mapping
            if user_id not in self.user_id_map or item_id not in self.item_id_map:
                continue
                
            # Get matrix indices
            user_idx = self.user_id_map[user_id]
            item_idx = self.item_id_map[item_id]
            
            # Determine interaction value
            if self.use_implicit:
                # For implicit feedback, just use 1.0 for engagement
                value = 1.0 if engaged else 0.0
            else:
                # For explicit feedback, could use rating or other value
                value = 1.0 if engaged else 0.0  # Simplified for this simulation
            
            # Only include positive interactions for implicit feedback
            if self.use_implicit and value <= 0:
                continue
                
            user_indices.append(user_idx)
            item_indices.append(item_idx)
            values.append(value)
        
        # If no interactions, skip training
        if not user_indices:
            logger.warning(f"{self.name}: No interactions to train on")
            return
        
        # Create sparse matrix
        interaction_matrix = csr_matrix(
            (values, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )
        
        # Calculate bias terms
        self.global_bias = np.mean(values) if values else 0.0
        
        # User bias: average deviation from global bias for each user
        self.user_bias = np.zeros(n_users)
        for user_idx in range(n_users):
            user_ratings = interaction_matrix[user_idx].data
            if len(user_ratings) > 0:
                self.user_bias[user_idx] = np.mean(user_ratings) - self.global_bias
                
        # Item bias: average deviation from global + user bias for each item
        self.item_bias = np.zeros(n_items)
        for item_idx in range(n_items):
            # Get all ratings for this item
            item_col = interaction_matrix.getcol(item_idx)
            if item_col.nnz > 0:
                # For each user who rated this item
                user_indices = item_col.indices
                item_ratings = item_col.data
                expected_ratings = np.array([self.global_bias + self.user_bias[u] for u in user_indices])
                self.item_bias[item_idx] = np.mean(item_ratings - expected_ratings)
        
        # Regularize bias terms
        self.user_bias *= (1.0 / (1.0 + self.regularization))
        self.item_bias *= (1.0 / (1.0 + self.regularization))
        
        # Remove biases from the matrix for better factorization
        if not self.use_implicit:
            # For explicit feedback, we can correct for biases
            interaction_matrix_copy = interaction_matrix.copy()
            for user_idx in range(n_users):
                user_row = interaction_matrix_copy.getrow(user_idx)
                if user_row.nnz > 0:
                    user_correction = self.global_bias + self.user_bias[user_idx]
                    item_indices = user_row.indices
                    item_correction = self.item_bias[item_indices]
                    interaction_matrix_copy.data[interaction_matrix_copy.indptr[user_idx]:
                                                 interaction_matrix_copy.indptr[user_idx+1]] -= (
                        user_correction + item_correction
                    )
            # Run SVD on the corrected matrix
            matrix_to_factorize = interaction_matrix_copy
        else:
            # For implicit feedback, we don't correct for biases
            matrix_to_factorize = interaction_matrix
        
        # Use TruncatedSVD to factorize the matrix
        try:
            # Skip factorization if we have too few interactions
            if matrix_to_factorize.nnz <= self.num_factors:
                # Create random factors instead
                logger.warning(f"{self.name}: Too few interactions for SVD, using random factors")
                self.user_factors = np.random.normal(0, 0.1, (n_users, self.num_factors))
                self.item_factors = np.random.normal(0, 0.1, (n_items, self.num_factors))
            else:
                # Perform truncated SVD
                svd = TruncatedSVD(n_components=self.num_factors, random_state=42)
                user_factors = svd.fit_transform(matrix_to_factorize)
                
                # Calculate item factors
                # V = U^T * Sigma
                singular_values = np.sqrt(svd.singular_values_)
                item_factors = svd.components_.T
                
                # Scale the factors by the singular values
                self.user_factors = user_factors
                self.item_factors = item_factors * singular_values
                
                # Calculate a simple measure of model quality: explained variance
                explained_variance = sum(svd.singular_values_) / matrix_to_factorize.nnz if matrix_to_factorize.nnz > 0 else 0
                self.add_metric('explained_variance', explained_variance)
                
        except Exception as e:
            logger.error(f"{self.name}: Error during SVD: {e}")
            # Create random factors as fallback
            self.user_factors = np.random.normal(0, 0.1, (n_users, self.num_factors))
            self.item_factors = np.random.normal(0, 0.1, (n_items, self.num_factors))
            
        # Log training time
        train_time = time.time() - start_time
        self.add_metric('train_time', train_time)
        self.training_count += 1
        
        logger.info(f"Trained {self.name} in {train_time:.4f} seconds")
        
    def recommend(self, 
                 user: User, 
                 items: List[Item], 
                 n: int = None,
                 exclude_items: Optional[Set[int]] = None) -> List[Item]:
        """
        Generate recommendations for a user based on collaborative filtering.
        
        For users seen during training, uses their learned factors.
        For new users, projects them into the latent space based on their interactions.
        
        Args:
            user: User to recommend items for
            items: Available items to recommend from
            n: Number of items to recommend
            exclude_items: Set of item IDs to exclude from recommendations
            
        Returns:
            List of recommended items sorted by predicted relevance
        """
        start_time = time.time()
        if self.user_factors is None and len(self.all_interactions) >= 20:
            # Force training if we have enough data but model not trained
            logger.info(f"{self.name}: Forcing training due to untrained model with sufficient data")
            self.train(interactions, items, users)

        # Default to self.top_k if n is not specified
        if n is None:
            n = self.top_k
            
        # Create set of items the user has already interacted with
        user_history_ids = {item.item_id for item in user.interaction_history}
        
        # Combine with additional items to exclude
        if exclude_items:
            exclude_ids = user_history_ids.union(exclude_items)
        else:
            exclude_ids = user_history_ids
            
        # Filter available items to those not in exclusion set
        available_items = self.filter_items(items, exclude_ids)
        
        # If no items are available after filtering, return empty list
        if not available_items:
            return []
        
        # If the model hasn't been trained yet, fall back to random recommendations
        if self.user_factors is None or self.item_factors is None:
            logger.warning(f"{self.name}: Model not trained, returning random recommendations")
            np.random.shuffle(available_items)
            return available_items[:n]
            
        # Get user's latent factor vector
        user_vector = self._get_user_vector(user)
        
        # If user vector is None (couldn't be computed), fall back to random
        if user_vector is None:
            np.random.shuffle(available_items)
            return available_items[:n]
            
        # Calculate predicted scores for all available items
        item_scores = []
        for item in available_items:
            item_id = item.item_id
            
            # If item not in our model, estimate score as 0
            if item_id not in self.item_id_map:
                item_scores.append((item, 0.0))
                continue
                
            item_idx = self.item_id_map[item_id]
            
            # Get item's latent factor vector
            item_vector = self.item_factors[item_idx]
            
            # Calculate base score as dot product of user and item vectors
            base_score = np.dot(user_vector, item_vector)
            
            # Add bias terms
            if not self.use_implicit:
                user_idx = self.user_id_map.get(user.user_id, 0)  # Default to 0 if user not found
                score = (
                    self.global_bias + 
                    self.user_bias[user_idx] + 
                    self.item_bias[item_idx] + 
                    base_score
                )
            else:
                # For implicit feedback, just use the dot product
                score = base_score
                
            item_scores.append((item, score))
            
        # Sort items by predicted score (descending)
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top n items
        recommendations = [item for item, _ in item_scores[:n]]
        
        # Update metrics
        inference_time = time.time() - start_time
        self.add_metric('inference_time', inference_time)
        
        return recommendations
        
    def _get_user_vector(self, user: User) -> Optional[np.ndarray]:
        """
        Get a user's latent factor vector.
        
        For users seen during training, returns their learned factors.
        For new users, projects them into the latent space based on their interactions.
        
        Args:
            user: User to get vector for
            
        Returns:
            User's latent factor vector or None if cannot be computed
        """
        # If user was in training set, return their learned factors
        if user.user_id in self.user_id_map:
            user_idx = self.user_id_map[user.user_id]
            return self.user_factors[user_idx]
            
        # For new users, project into latent space based on their interactions
        # This is a simplified implementation of the folding-in technique
        
        # If user has no interactions, return None
        if not user.interaction_history:
            return None
            
        # Get item vectors for items the user has interacted with
        item_vectors = []
        for item in user.interaction_history:
            item_id = item.item_id
            if item_id in self.item_id_map:
                item_idx = self.item_id_map[item_id]
                item_vectors.append(self.item_factors[item_idx])
                
        # If no valid item vectors, return None
        if not item_vectors:
            return None
            
        # Average the item vectors
        user_vector = np.mean(item_vectors, axis=0)
        
        return user_vector
        
    def get_most_similar_items(self, item: Item, n: int = 10, all_items: List[Item] = None) -> List[Tuple[Item, float]]:
        """
        Find items most similar to a given item based on latent factors.
        
        Args:
            item: Reference item
            n: Number of similar items to return
            all_items: List of all items to consider
            
        Returns:
            List of (item, similarity) tuples
        """
        # If model hasn't been trained or item not in model, return empty list
        if (self.item_factors is None or 
            item.item_id not in self.item_id_map):
            return []
            
        # Need the list of all items to lookup by ID
        if all_items is None:
            return []
            
        # Get item's latent factor vector
        item_idx = self.item_id_map[item.item_id]
        item_vector = self.item_factors[item_idx]
        
        # Calculate similarity to all other items
        similarities = []
        for i, other_id in self.reverse_item_map.items():
            # Skip if same item
            if other_id == item.item_id:
                continue
                
            other_vector = self.item_factors[i]
            
            # Calculate cosine similarity
            similarity = np.dot(item_vector, other_vector) / (
                np.linalg.norm(item_vector) * np.linalg.norm(other_vector)
            )
            
            similarities.append((other_id, similarity))
            
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top n
        top_similar = similarities[:n]
        
        # Convert item IDs back to Item objects
        # This requires looking up in the provided all_items list
        result = []
        for item_id, sim in top_similar:
            # Find the item with this ID
            matching_item = next((it for it in all_items if it.item_id == item_id), None)
            if matching_item is not None:
                result.append((matching_item, sim))
        
        return result
        
    def get_most_similar_users(self, user: User, users: List[User], n: int = 10) -> List[Tuple[User, float]]:
        """
        Find users most similar to a given user based on latent factors.
        
        Args:
            user: Reference user
            users: List of all users
            n: Number of similar users to return
            
        Returns:
            List of (user, similarity) tuples
        """
        # If model hasn't been trained or user not in model, return empty list
        if self.user_factors is None:
            return []
            
        # Get user's latent factor vector
        user_vector = self._get_user_vector(user)
        
        if user_vector is None:
            return []
            
        # Calculate similarity to all other users
        similarities = []
        for other_user in users:
            # Skip if same user
            if other_user.user_id == user.user_id:
                continue
                
            other_vector = self._get_user_vector(other_user)
            
            # Skip if other user has no vector
            if other_vector is None:
                continue
                
            # Calculate cosine similarity
            similarity = np.dot(user_vector, other_vector) / (
                np.linalg.norm(user_vector) * np.linalg.norm(other_vector)
            )
            
            similarities.append((other_user, similarity))
            
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top n
        return similarities[:n]