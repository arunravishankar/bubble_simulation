from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Optional, Set
from ..content.item import Item
from ..users.user import User
from ..settings import (
    RECO_TOP_K
)

logger = logging.getLogger(__name__)

class BaseRecommender(ABC):
    """
    Abstract base class for recommendation algorithms.
    
    All recommendation models should inherit from this class and implement
    the required methods for training and generating recommendations.
    """
    
    def __init__(self, 
                 name: str = "BaseRecommender",
                 retrain_frequency: int = 10,
                 top_k: int = RECO_TOP_K):
        """
        Initialize the recommender.
        
        Args:
            name: Identifier for this recommender instance
            retrain_frequency: How often to retrain the model (in timesteps)
            top_k: Default number of items to recommend
        """
        self.name = name
        self.retrain_frequency = retrain_frequency
        self.top_k = top_k
        self.last_training_step = -1
        self.training_count = 0
        
        # Metrics tracking
        self.metrics = {
            'train_time': [],
            'inference_time': [],
            'train_loss': []
        }
    
    @abstractmethod
    def train(self, interactions: List[Dict], items: List[Item], users: List[User]) -> None:
        """
        Train the recommendation model.
        
        Args:
            interactions: List of user-item interaction dictionaries
            items: List of available content items
            users: List of users
        """
        pass
    
    @abstractmethod
    def recommend(self, 
                 user: User, 
                 items: List[Item], 
                 n: int = None,
                 exclude_items: Optional[Set[int]] = None) -> List[Item]:
        """
        Generate recommendations for a specific user.
        
        Args:
            user: User to recommend items for
            items: Available items to recommend from
            n: Number of items to recommend, defaults to self.top_k
            exclude_items: Set of item IDs to exclude from recommendations
            
        Returns:
            List of recommended items
        """
        pass
    
    def update(self, 
              interactions: List[Dict], 
              items: List[Item], 
              users: List[User],
              timestep: int) -> bool:
        """
        Update the model with new data, retraining if needed based on frequency.
        
        Args:
            interactions: List of user-item interaction dictionaries
            items: List of available content items
            users: List of users
            timestep: Current simulation timestep
            
        Returns:
            Boolean indicating whether the model was retrained
        """
        # Check if we need to retrain
        should_retrain = (self.last_training_step == -1 or  # First time
                         timestep - self.last_training_step >= self.retrain_frequency or
                         len(interactions) >= 50)
        
        if should_retrain:
            logger.info(f"Retraining {self.name} at timestep {timestep}")
            self.train(interactions, items, users)
            self.last_training_step = timestep
            return True
        
        return False
    
    def filter_items(self, 
                   items: List[Item], 
                   exclude_item_ids: Optional[Set[int]] = None) -> List[Item]:
        """
        Filter items based on exclusion criteria.
        
        Args:
            items: List of items to filter
            exclude_item_ids: Set of item IDs to exclude
            
        Returns:
            Filtered list of items
        """
        if exclude_item_ids is None:
            return items
            
        return [item for item in items if item.item_id not in exclude_item_ids]
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Get performance metrics for this recommender.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()
    
    def add_metric(self, metric_name: str, value: float) -> None:
        """
        Add a value to a specific metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to add
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            
        self.metrics[metric_name].append(value)
    
    def _get_user_item_interactions(self, 
                                  interactions: List[Dict]) -> Dict[int, Dict[int, float]]:
        """
        Convert interaction list to user->item->engagement mapping.
        
        Args:
            interactions: List of interaction dictionaries
            
        Returns:
            Nested dictionary mapping user ID to item ID to engagement score
        """
        user_item_interactions = {}
        
        for interaction in interactions:
            user_id = interaction['user_id']
            item_id = interaction['item_id']
            engaged = interaction['engaged']
            
            if user_id not in user_item_interactions:
                user_item_interactions[user_id] = {}
                
            # If user engaged, use 1.0 as score, otherwise 0.0
            user_item_interactions[user_id][item_id] = 1.0 if engaged else 0.0
            
        return user_item_interactions