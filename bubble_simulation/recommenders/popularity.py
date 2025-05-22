import time
import logging
from typing import List, Dict, Optional, Set
from ..content.item import Item
from ..users.user import User
from .base import BaseRecommender
from ..settings import (
    RECO_TOP_K
)

logger = logging.getLogger(__name__)

class PopularityRecommender(BaseRecommender):
    """
    Simple recommender that recommends items based on their popularity scores.
    
    This recommender serves as a baseline that doesn't personalize recommendations
    but simply recommends the most popular items that a user hasn't seen yet.
    """
    
    def __init__(self, 
                 name: str = "PopularityRecommender",
                 retrain_frequency: int = 10,
                 top_k: int = RECO_TOP_K,
                 recency_weight: float = 0.7):
        """
        Initialize the popularity recommender.
        
        Args:
            name: Identifier for this recommender instance
            retrain_frequency: How often to recalculate popularity (in timesteps)
            top_k: Default number of items to recommend
            recency_weight: Weight given to recent interactions vs. intrinsic popularity
        """
        super().__init__(name=name, retrain_frequency=retrain_frequency, top_k=top_k)
        self.recency_weight = recency_weight
        self.item_popularity = {}  # Item ID to popularity score mapping
        self.interaction_counts = {}  # Item ID to interaction count mapping
        
    def train(self, interactions: List[Dict], items: List[Item], users: List[User]) -> None:
        """
        Calculate item popularities based on interaction counts and intrinsic popularity.
        
        Args:
            interactions: List of user-item interaction dictionaries
            items: List of available content items
            users: List of users
        """
        start_time = time.time()
        
        # Reset or initialize interaction counts if first training
        if not self.interaction_counts:
            self.interaction_counts = {item.item_id: 0 for item in items}
        
        # Count recent interactions
        for interaction in interactions:
            item_id = interaction['item_id']
            engaged = interaction['engaged']
            
            # Only count interactions where the user engaged
            if engaged and item_id in self.interaction_counts:
                self.interaction_counts[item_id] += 1
        
        # Calculate normalized popularity scores by combining:
        # 1. Interaction counts (engagement popularity)
        # 2. Intrinsic popularity (from item.popularity_score)
        
        # Normalize interaction counts to [0, 1] range
        max_interactions = max(self.interaction_counts.values()) if self.interaction_counts.values() else 1
        normalized_interactions = {
            item_id: count / max_interactions if max_interactions > 0 else 0
            for item_id, count in self.interaction_counts.items()
        }
        
        # Combine interaction counts with intrinsic popularity
        self.item_popularity = {}
        for item in items:
            interaction_popularity = normalized_interactions.get(item.item_id, 0)
            intrinsic_popularity = item.popularity_score
            
            # Weighted combination
            self.item_popularity[item.item_id] = (
                self.recency_weight * interaction_popularity + 
                (1 - self.recency_weight) * intrinsic_popularity
            )
        
        # Update metrics
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
        Recommend the most popular items that the user hasn't seen.
        
        Args:
            user: User to recommend items for
            items: Available items to recommend from
            n: Number of items to recommend
            exclude_items: Set of item IDs to exclude from recommendations
            
        Returns:
            List of recommended items sorted by popularity
        """
        start_time = time.time()
        
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
            
        # Sort items by popularity score (descending)
        sorted_items = sorted(
            available_items,
            key=lambda item: self.item_popularity.get(item.item_id, 0),
            reverse=True
        )
        
        # Take top n items
        recommendations = sorted_items[:n]
        
        # Update metrics
        inference_time = time.time() - start_time
        self.add_metric('inference_time', inference_time)
        
        return recommendations
    
    def get_item_popularity(self, item_id: int) -> float:
        """
        Get the calculated popularity score for an item.
        
        Args:
            item_id: The item identifier
            
        Returns:
            Popularity score, or 0 if not found
        """
        return self.item_popularity.get(item_id, 0.0)