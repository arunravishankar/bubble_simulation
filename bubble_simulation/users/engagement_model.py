from dataclasses import dataclass
import numpy as np
from typing import List, Dict
from ..content.item import Item
from .user import User
from ..settings import (
    ENGAGEMENT_BASE_RATE, ENGAGEMENT_POSITION_DECAY, ENGAGEMENT_NOVELTY_FACTOR,
    ENGAGEMENT_DIVERSITY_FACTOR, ENGAGEMENT_TIME_PENALTY,
    RECENT_HISTORY_SIZE
)


@dataclass
class EngagementModel:
    """
    Models how users engage with recommended content.
    
    This class provides functions to calculate engagement probabilities
    based on user preferences, content features, position, and context.
    """
    
    base_engagement_rate: float = ENGAGEMENT_BASE_RATE
    position_decay_factor: float = ENGAGEMENT_POSITION_DECAY
    novelty_factor: float = ENGAGEMENT_NOVELTY_FACTOR
    category_diversity_factor: float = ENGAGEMENT_DIVERSITY_FACTOR
    time_penalty_factor: float = ENGAGEMENT_TIME_PENALTY
    
    def calculate_engagement_probability(self, 
                                         user: User, 
                                         item: Item, 
                                         position: int,
                                         item_history: List[Item]) -> float:
        """
        Calculate probability that a user will engage with an item.
        
        Args:
            user: The user being recommended to
            item: The recommended item
            position: Position in the recommendation list (0-indexed)
            item_history: User's recent interaction history
            
        Returns:
            Probability of engagement (0.0 to 1.0)
        """
        # Start with preference-based probability
        base_prob = user.get_preference_for_item(item)
        
        # Apply position bias
        position_discount = self.calculate_position_bias(position, user.position_bias_factor)
        
        # Calculate novelty impact (positive for users who like exploration)
        novelty_impact = self.calculate_novelty_impact(item, item_history)
        
        # Calculate diversity bonus
        diversity_bonus = self.calculate_diversity_bonus(item, item_history)
        
        # Combine factors into final probability
        # Users who prefer exploration get more novelty impact
        engagement_prob = (
            base_prob * position_discount + 
            novelty_impact * user.exploration_factor +
            diversity_bonus * user.diversity_preference
        )
        
        # Ensure probability is in [0, 1] range
        return max(0.0, min(1.0, engagement_prob))
    
    def calculate_position_bias(self, position: int, user_position_factor: float) -> float:
        """
        Calculate the effect of position on engagement probability.
        
        Args:
            position: Position in recommendation list (0-indexed)
            user_position_factor: User's susceptibility to position bias
            
        Returns:
            Position discount factor
        """
        # Exponential decay of attention with position
        return np.power(self.position_decay_factor, position * user_position_factor)
    
    def calculate_novelty_impact(self, item: Item, item_history: List[Item]) -> float:
        """
        Calculate how item novelty affects engagement.
        
        Args:
            item: The candidate item
            item_history: User's interaction history
            
        Returns:
            Novelty impact factor
        """
        if not item_history:
            return self.novelty_factor  # Maximum novelty for new users
            
        # Limit to recent history
        recent_history = item_history[-RECENT_HISTORY_SIZE:] if len(item_history) > RECENT_HISTORY_SIZE else item_history
        
        # Check if this item's categories have been seen before
        seen_categories = set()
        for hist_item in recent_history:
            seen_categories.update(hist_item.categories)
            
        # Calculate category novelty
        item_categories = set(item.categories)
        new_categories = item_categories - seen_categories
        
        if not item_categories:  # Handle items with no categories
            category_novelty = 0.0
        else:
            category_novelty = len(new_categories) / len(item_categories)
            
        # Calculate feature novelty (average distance to history items)
        if recent_history:
            similarities = [item.similarity(hist_item) for hist_item in recent_history]
            avg_similarity = sum(similarities) / len(similarities)
            feature_novelty = 1.0 - avg_similarity
        else:
            feature_novelty = 1.0
            
        # Combine category and feature novelty
        return self.novelty_factor * (0.5 * category_novelty + 0.5 * feature_novelty)
    
    def calculate_diversity_bonus(self, item: Item, item_history: List[Item]) -> float:
        """
        Calculate diversity bonus for recommending this item.
        
        Args:
            item: The candidate item
            item_history: User's interaction history
            
        Returns:
            Diversity bonus factor
        """
        if not item_history or not item.categories:
            return 0.0
            
        # Count category frequencies in history
        category_counts = {}
        for hist_item in item_history:
            for category in hist_item.categories:
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1
                
        # Calculate inverse frequency for item's categories
        total_categories = sum(category_counts.values())
        inverse_frequencies = []
        
        for category in item.categories:
            if category in category_counts:
                freq = category_counts[category] / total_categories
                inverse_frequencies.append(1.0 - freq)
            else:
                # Maximum diversity for unseen categories
                inverse_frequencies.append(1.0)
                
        # Average inverse frequency is the diversity bonus
        if inverse_frequencies:
            return self.category_diversity_factor * sum(inverse_frequencies) / len(inverse_frequencies)
        return 0.0
    
    def calculate_user_affinity(self, user: User, item: Item) -> float:
        """
        Calculate a user's affinity for an item regardless of context.
        
        Useful for initial model training data generation.
        
        Args:
            user: The user
            item: The item
            
        Returns:
            Affinity score (higher means stronger affinity)
        """
        # Base affinity is preference score
        affinity = user.get_preference_for_item(item)
        
        # Include some randomness
        random_factor = np.random.normal(0, 0.1)
        
        # Combine with a slight random variation
        return max(0.0, min(1.0, affinity + random_factor))
    
    def generate_engagement_events(self, 
                                 user: User, 
                                 items: List[Item], 
                                 num_events: int) -> List[Dict]:
        """
        Generate synthetic engagement events for initial model training.
        
        Args:
            user: The user
            items: List of available items
            num_events: Number of events to generate
            
        Returns:
            List of engagement events
        """
        events = []
        
        # Select random items weighted by affinity
        affinities = [self.calculate_user_affinity(user, item) for item in items]
        
        # Ensure non-negative affinities for sampling
        min_affinity = min(affinities)
        if min_affinity < 0:
            affinities = [a - min_affinity + 0.01 for a in affinities]
            
        # Normalize to create a probability distribution
        total_affinity = sum(affinities)
        if total_affinity > 0:
            probabilities = [a / total_affinity for a in affinities]
        else:
            # If all affinities are 0, use uniform distribution
            probabilities = [1.0 / len(items) for _ in items]
            
        # Sample items according to probabilities
        sample_size = min(num_events, len(items))

        if sample_size == 0 or len(items) == 0:
            selected_indices = []
        else:
            # Ensure probabilities sum to 1 and handle edge cases
            probabilities = np.array(probabilities)
            
            # Handle case where all probabilities are 0
            if np.sum(probabilities) == 0:
                probabilities = np.ones(len(items)) / len(items)  # Uniform distribution
            else:
                probabilities = probabilities / np.sum(probabilities)  # Normalize
            
            # Check if we can sample without replacement
            if sample_size <= len(items):
                try:
                    selected_indices = np.random.choice(
                        len(items), 
                        size=sample_size, 
                        replace=False, 
                        p=probabilities
                    )
                except ValueError:
                    # Fallback: sample with replacement if without replacement fails
                    selected_indices = np.random.choice(
                        len(items), 
                        size=sample_size, 
                        replace=True, 
                        p=probabilities
                    )
            else:
                selected_indices = []
        
        # Create engagement events
        for i, idx in enumerate(selected_indices):
            item = items[idx]
            position = i % user.attention_span  # Assign positions based on user's attention span
            
            # Probability of engagement decreases with position
            engagement_prob = self.calculate_engagement_probability(
                user, item, position, user.interaction_history
            )
            
            # Determine if user engaged
            engaged = np.random.random() < engagement_prob
            
            # Create event
            event = {
                'user_id': user.user_id,
                'item_id': item.item_id,
                'position': position,
                'engaged': engaged,
                'timestamp': i,
                'timestep': 0
            }
            
            events.append(event)
            
            # If engaged, update user history and preferences
            if engaged:
                user.interaction_history.append(item)
                user.update_preferences(item, engaged=True)
                
        return events