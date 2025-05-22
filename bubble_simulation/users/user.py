from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from ..content.item import Item
from ..settings import (
    USER_PREFERENCE_ADAPTATION_RATE, USER_PREFERENCE_FEATURE_WEIGHT,
    USER_PREFERENCE_CATEGORY_WEIGHT, USER_ATTENTION_SPAN,
    USER_EXPLORATION_FACTOR_MEAN, USER_POSITION_BIAS_FACTOR_MEAN,
    USER_DIVERSITY_PREFERENCE_MEAN
)


@dataclass
class User:
    """
    Represents a user in the recommendation system simulation.
    
    A user has preferences, interaction history, and engagement properties
    that determine how they respond to recommendations.
    """
    user_id: int
    preference_vector: np.ndarray
    exploration_factor: float = USER_EXPLORATION_FACTOR_MEAN # Willingness to explore new content
    position_bias_factor: float = USER_POSITION_BIAS_FACTOR_MEAN # How much position affects engagement
    diversity_preference: float = USER_DIVERSITY_PREFERENCE_MEAN # Preference for diverse recommendations
    attention_span: int = USER_ATTENTION_SPAN # How many recommended items they typically consider    
    # These will be initialized in post_init
    interaction_history: List[Item] = field(default_factory=list)
    category_interests: Dict[int, float] = field(default_factory=dict)
    engagement_counts: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize additional attributes after dataclass initialization."""
        # Initialize engagement counts
        self.engagement_counts = {
            'total_recommended': 0,
            'total_engaged': 0,
            'last_session_recommended': 0,
            'last_session_engaged': 0
        }
        
    def engage_with_item(self, item: Item, position: int, engagement_model) -> bool:
        """
        Determine if the user engages with a recommended item.
        
        Args:
            item: The recommended content item
            position: Position in recommendation list (0-indexed)
            engagement_model: Model that calculates engagement probability
            
        Returns:
            Boolean indicating whether the user engaged
        """
        # Increment recommendation counts
        self.engagement_counts['total_recommended'] += 1
        self.engagement_counts['last_session_recommended'] += 1
        
        # Calculate engagement probability using the engagement model
        engagement_prob = engagement_model.calculate_engagement_probability(
            user=self,
            item=item,
            position=position,
            item_history=self.interaction_history
        )
        
        # Determine engagement based on probability
        engaged = np.random.random() < engagement_prob
        
        # If engaged, update interaction history and preferences
        if engaged:
            self.interaction_history.append(item)
            self.engagement_counts['total_engaged'] += 1
            self.engagement_counts['last_session_engaged'] += 1
            self.update_preferences(item, engaged=True)
            
        return engaged
    
    def update_preferences(self, item: Item, engaged: bool) -> None:
        """
        Update user preferences based on interaction with an item.
        
        Args:
            item: The item the user interacted with
            engaged: Whether the user engaged positively
        """
        # Update preference vector through a weighted average
        # The more a user interacts with similar content, the more their 
        # preferences align with that content
        adaptation_rate = USER_PREFERENCE_ADAPTATION_RATE # How quickly preferences adapt
        if engaged:
            # Move preference vector slightly toward item features
            self.preference_vector = (1 - adaptation_rate) * self.preference_vector + \
                                    adaptation_rate * item.features
            
            # Normalize preference vector
            norm = np.linalg.norm(self.preference_vector)
            if norm > 0:
                self.preference_vector = self.preference_vector / norm
                
            # Update category interests
            for category in item.categories:
                if category not in self.category_interests:
                    self.category_interests[category] = 0.0
                # Increase interest in this category
                self.category_interests[category] = min(
                    1.0, 
                    self.category_interests[category] + adaptation_rate
                )
                
    def get_preference_for_item(self, item: Item) -> float:
        """
        Calculate the user's preference score for a specific item.
        
        Args:
            item: The item to evaluate
            
        Returns:
            Preference score (higher means stronger preference)
        """
        # Calculate base preference using cosine similarity
        preference = np.dot(self.preference_vector, item.features)
        
        # Adjust based on category interests
        category_bonus = 0
        for category in item.categories:
            category_bonus += self.category_interests.get(category, 0.0)
            
        # Normalize category bonus
        if item.categories:
            category_bonus /= len(item.categories)
            
        # Combine base preference with category bonus
        return USER_PREFERENCE_FEATURE_WEIGHT * preference + USER_PREFERENCE_CATEGORY_WEIGHT * category_bonus

    
    def start_new_session(self) -> None:
        """Reset session-level engagement counts when starting a new session."""
        self.engagement_counts['last_session_recommended'] = 0
        self.engagement_counts['last_session_engaged'] = 0
        
    def get_engagement_rate(self) -> float:
        """
        Calculate the user's overall engagement rate.
        
        Returns:
            Ratio of engaged items to recommended items
        """
        if self.engagement_counts['total_recommended'] == 0:
            return 0.0
        return self.engagement_counts['total_engaged'] / self.engagement_counts['total_recommended']
    
    def get_category_preferences(self) -> Dict[int, float]:
        """
        Get the user's preference distribution across categories.
        
        Returns:
            Dictionary of category IDs to preference scores
        """
        return self.category_interests.copy()
    
    def get_diversity_metric(self) -> float:
        """
        Calculate the diversity of the user's recent interactions.
        
        Returns:
            Diversity score (higher means more diverse)
        """
        # If no interactions, return 0
        if not self.interaction_history:
            return 0.0
            
        # Limit to the most recent items
        recent_history = self.interaction_history[-20:]
        
        # Count unique categories
        categories = set()
        for item in recent_history:
            categories.update(item.categories)
            
        # Diversity is the ratio of unique categories to total interactions
        # scaled by the total number of categories
        return len(categories) / (len(recent_history) + 1)  # Add 1 to avoid division by zero