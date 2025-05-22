from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Any


@dataclass
class Item:
    """
    Represents a single content item in the recommendation universe.
    
    An item has features, categories, and metadata that the
    recommendation system uses to make decisions.
    """
    item_id: int
    features: np.ndarray
    categories: List[int] = field(default_factory=list)
    popularity_score: float = 0.0
    creation_time: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.item_id)  # Use item_id as the hash
        
    def __eq__(self, other):
        if not isinstance(other, Item):
            return False
        return self.item_id == other.item_id
    
    def similarity(self, other_item: 'Item') -> float:
        """
        Calculate similarity between this item and another item using cosine similarity.
        
        Args:
            other_item: Item to compare with
            
        Returns:
            Similarity score (higher means more similar)
        """
        if len(self.features) == 0 or len(other_item.features) == 0:
            return 0.0
            
        dot_product = np.dot(self.features, other_item.features)
        norm_self = np.linalg.norm(self.features)
        norm_other = np.linalg.norm(other_item.features)
        
        # Avoid division by zero
        if norm_self == 0 or norm_other == 0:
            return 0.0
            
        return dot_product / (norm_self * norm_other)