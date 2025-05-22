from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from ..content.content_universe import ContentUniverse
from .user import User
from .engagement_model import EngagementModel
from ..settings import (
    USER_NUM_USERS, USER_NUM_FEATURES, USER_EXPLORATION_FACTOR_MEAN,
    USER_EXPLORATION_FACTOR_STD, USER_POSITION_BIAS_FACTOR_MEAN,
    USER_POSITION_BIAS_FACTOR_STD, USER_DIVERSITY_PREFERENCE_MEAN,
    USER_DIVERSITY_PREFERENCE_STD, SIM_INITIAL_INTERACTIONS_PER_USER,
    DEFAULT_SEED
)


@dataclass
class UserUniverse:
    """
    Represents the collection of users in the simulation.
    
    This class manages user creation, preference initialization,
    and provides utilities for user analysis.
    """
    
    num_users: int = USER_NUM_USERS
    num_user_features: int = USER_NUM_FEATURES
    exploration_factor_mean: float = USER_EXPLORATION_FACTOR_MEAN
    exploration_factor_std: float = USER_EXPLORATION_FACTOR_STD
    position_bias_factor_mean: float = USER_POSITION_BIAS_FACTOR_MEAN
    position_bias_factor_std: float = USER_POSITION_BIAS_FACTOR_STD
    diversity_preference_mean: float = USER_DIVERSITY_PREFERENCE_MEAN
    diversity_preference_std: float = USER_DIVERSITY_PREFERENCE_STD
    seed: Optional[int] = DEFAULT_SEED
    
    # These will be initialized in post_init
    users: List[User] = field(default_factory=list)
    user_id_map: Dict[int, User] = field(default_factory=dict)
    engagement_model: EngagementModel = field(default_factory=EngagementModel)
    rng: np.random.Generator = field(init=False)
    
    def __post_init__(self):
        """Initialize additional attributes after dataclass initialization."""
        self.rng = np.random.default_rng(self.seed)
        
    def generate_users(self, content_universe: ContentUniverse) -> None:
        """
        Generate the user universe with diverse user preferences.
        
        Args:
            content_universe: The content catalog used to initialize preferences
        """
        # Clear existing users if any
        self.users = []
        self.user_id_map = {}
        
        # Create user preference clusters
        # Users will have preferences that align with different content categories
        num_clusters = min(10, content_universe.num_categories)
        
        # Get category centers from content universe to align preferences
        category_centers = []
        for category_id in range(content_universe.num_categories):
            category_items = content_universe.get_items_by_category(category_id)
            if category_items:
                # Use average feature vector as center
                features = [item.features for item in category_items]
                center = np.mean(features, axis=0)
                # Normalize
                norm = np.linalg.norm(center)
                if norm > 0:
                    center = center / norm
                category_centers.append(center)
        
        # If we couldn't get enough category centers, create random ones
        while len(category_centers) < num_clusters:
            center = self.rng.normal(0, 1, self.num_user_features)
            center = center / np.linalg.norm(center)
            category_centers.append(center)
        
        # Assign users to preference clusters
        users_per_cluster = [self.num_users // num_clusters] * num_clusters
        # Distribute remaining users
        remainder = self.num_users - sum(users_per_cluster)
        for i in range(remainder):
            users_per_cluster[i] += 1
            
        # Create users
        user_id = 0
        for cluster_id, cluster_size in enumerate(users_per_cluster):
            cluster_center = category_centers[cluster_id % len(category_centers)]
            
            for _ in range(cluster_size):
                # Generate preference vector as cluster center + noise
                preference_vector = cluster_center + self.rng.normal(0, 0.3, self.num_user_features)
                # Normalize
                preference_vector = preference_vector / np.linalg.norm(preference_vector)
                
                # Sample user traits from distributions
                exploration_factor = self.rng.normal(
                    self.exploration_factor_mean, 
                    self.exploration_factor_std
                )
                exploration_factor = max(0.01, min(0.99, exploration_factor))
                
                position_bias_factor = self.rng.normal(
                    self.position_bias_factor_mean, 
                    self.position_bias_factor_std
                )
                position_bias_factor = max(0.01, min(0.99, position_bias_factor))
                
                diversity_preference = self.rng.normal(
                    self.diversity_preference_mean, 
                    self.diversity_preference_std
                )
                diversity_preference = max(0.01, min(0.99, diversity_preference))
                
                # Create user
                user = User(
                    user_id=user_id,
                    preference_vector=preference_vector,
                    exploration_factor=exploration_factor,
                    position_bias_factor=position_bias_factor,
                    diversity_preference=diversity_preference
                )
                
                self.users.append(user)
                self.user_id_map[user_id] = user
                
                user_id += 1
                
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Retrieve a user by ID.
        
        Args:
            user_id: The user identifier
            
        Returns:
            The requested user or None if not found
        """
        return self.user_id_map.get(user_id)
    
    def generate_initial_interactions(self, content_universe: ContentUniverse, 
                                      interactions_per_user: int = SIM_INITIAL_INTERACTIONS_PER_USER
                                    ) -> List[Dict]:

        """
        Generate initial user-item interactions to bootstrap the system.
        
        Args:
            content_universe: The content catalog
            interactions_per_user: Number of interactions to generate per user
            
        Returns:
            List of interaction events
        """
        all_events = []
        
        # Generate interactions for each user
        for user in self.users:
            # Get all items
            items = content_universe.items
            
            # Generate events
            user_events = self.engagement_model.generate_engagement_events(
                user=user,
                items=items,
                num_events=interactions_per_user
            )
            
            all_events.extend(user_events)
            
        return all_events
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the user universe to a pandas DataFrame for analysis.
        
        Returns:
            DataFrame with user features and properties
        """
        data = {
            'user_id': [],
            'exploration_factor': [],
            'position_bias_factor': [],
            'diversity_preference': [],
            'num_interactions': [],
            'engagement_rate': []
        }
        
        # Add feature columns
        for i in range(self.num_user_features):
            data[f'preference_{i}'] = []
            
        # Add category interests
        all_categories = set()
        for user in self.users:
            all_categories.update(user.category_interests.keys())
            
        for category in all_categories:
            data[f'category_interest_{category}'] = []
            
        # Populate data
        for user in self.users:
            data['user_id'].append(user.user_id)
            data['exploration_factor'].append(user.exploration_factor)
            data['position_bias_factor'].append(user.position_bias_factor)
            data['diversity_preference'].append(user.diversity_preference)
            data['num_interactions'].append(len(user.interaction_history))
            data['engagement_rate'].append(user.get_engagement_rate())
            
            for i, value in enumerate(user.preference_vector):
                data[f'preference_{i}'].append(value)
                
            for category in all_categories:
                data[f'category_interest_{category}'].append(
                    user.category_interests.get(category, 0.0)
                )
                
        return pd.DataFrame(data)
    
    def get_user_similarity(self, user1: User, user2: User) -> float:
        """
        Calculate similarity between two users based on preferences.
        
        Args:
            user1: First user
            user2: Second user
            
        Returns:
            Similarity score (higher means more similar)
        """
        # Cosine similarity between preference vectors
        dot_product = np.dot(user1.preference_vector, user2.preference_vector)
        norm1 = np.linalg.norm(user1.preference_vector)
        norm2 = np.linalg.norm(user2.preference_vector)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def get_similar_users(self, user: User, n: int = 10) -> List[User]:
        """
        Find users similar to the given user.
        
        Args:
            user: Reference user
            n: Number of similar users to return
            
        Returns:
            List of similar users
        """
        # Calculate similarity to all other users
        similarities = [(u, self.get_user_similarity(user, u)) 
                    for u in self.users if u.user_id != user.user_id]
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n
        return [u for u, _ in similarities[:n]]