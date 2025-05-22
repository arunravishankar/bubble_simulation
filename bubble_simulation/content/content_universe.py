from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set
from .item import Item
from ..settings import (
    CONTENT_NUM_ITEMS, CONTENT_NUM_CATEGORIES, CONTENT_NUM_FEATURES,
    CONTENT_POPULARITY_POWER_LAW, CONTENT_GROWTH_RATE,
    CONTENT_FEATURE_NOISE, CONTENT_MULTI_CATEGORY_PROBABILITY,
    DEFAULT_SEED
)

@dataclass
class ContentUniverse:
    """
    Represents the entire content catalog available for recommendation.
    
    This class manages a collection of content items with their features,
    categories, and popularity distributions. It can also model dynamic
    content catalogs where new items are added over time.
    """
    
    num_items: int = CONTENT_NUM_ITEMS
    num_categories: int = CONTENT_NUM_CATEGORIES
    num_features: int = CONTENT_NUM_FEATURES
    popularity_power_law: float = CONTENT_POPULARITY_POWER_LAW
    dynamic_content: bool = False
    content_growth_rate: float = CONTENT_GROWTH_RATE
    seed: Optional[int] = DEFAULT_SEED
    
    # These will be initialized in post_init
    items: List[Item] = field(default_factory=list)
    category_items: Dict[int, Set[Item]] = field(default_factory=dict)
    item_id_map: Dict[int, Item] = field(default_factory=dict)
    next_item_id: int = field(init=False)
    rng: np.random.Generator = field(init=False)
    
    def __post_init__(self):
        """Initialize additional attributes after dataclass initialization."""
        self.next_item_id = self.num_items
        self.rng = np.random.default_rng(self.seed)
        self.category_items = {i: set() for i in range(self.num_categories)}
        
    def generate_content(self) -> None:
        """
        Generate the content universe with items, features, and categories.
        
        This creates all items with appropriate feature vectors and assigns
        them to categories based on configurable distributions.
        """
        # Clear existing items if any
        self.items = []
        self.item_id_map = {}
        self.category_items = {i: set() for i in range(self.num_categories)}
        
        # Generate category centers (to make items within categories more similar)
        category_centers = self.rng.normal(0, 1, (self.num_categories, self.num_features))
        
        ## PRODUCE MORE NOISE HERE INSTEAD OF GETTING EXACTLY A POWER LAW
        # Generate popularity distribution following power law
        popularity_scores = np.power(
            np.arange(1, self.num_items + 1, dtype=float), -self.popularity_power_law
        )
        # Normalize to [0, 1] range
        popularity_scores = popularity_scores / np.sum(popularity_scores)
        
        # Shuffle to avoid correlation between ID and popularity
        self.rng.shuffle(popularity_scores)
        
        # Assign items to categories with some randomness
        category_sizes = self._generate_category_sizes()
        
        # Create items
        item_id = 0
        for category_id, size in enumerate(category_sizes):
            category_center = category_centers[category_id]
            
            for _ in range(size):
                if item_id >= self.num_items:
                    break
                    
                # Generate item features as category center + noise
                features = category_center + self.rng.normal(0, CONTENT_FEATURE_NOISE, self.num_features)

                # Normalize features
                features = features / np.linalg.norm(features)
                
                # Assign primary category + possible additional categories
                num_categories = 1 + self.rng.binomial(2, CONTENT_MULTI_CATEGORY_PROBABILITY)
                categories = [category_id]  # Primary category
                if num_categories > 1:
                    # Add additional random categories
                    possible_additional = [c for c in range(self.num_categories) if c != category_id]
                    if possible_additional:
                        additional_categories = self.rng.choice(
                            possible_additional,
                            size=min(num_categories-1, len(possible_additional)),
                            replace=False
                        )
                        categories.extend(additional_categories)
                
                item = Item(
                    item_id=item_id,
                    features=features,
                    categories=categories,
                    popularity_score=popularity_scores[item_id],
                    creation_time=0  # Initial items all created at time 0
                )
                
                self.items.append(item)
                self.item_id_map[item_id] = item
                for cat in categories:
                    self.category_items[cat].add(item)
                
                item_id += 1
    
    def _generate_category_sizes(self) -> List[int]:
        """
        Generate the number of items per category following a distribution.
        
        Returns:
            List of category sizes
        """
        # Use a Dirichlet distribution to generate category size proportions
        alpha = [1.0] * self.num_categories
        proportions = self.rng.dirichlet(alpha)
        
        # Convert to actual sizes
        sizes = np.round(proportions * self.num_items).astype(int)
        
        # Ensure we have exactly num_items total
        diff = self.num_items - np.sum(sizes)
        if diff > 0:
            # Add remaining items to random categories
            indices = self.rng.choice(self.num_categories, size=diff, replace=True)
            for idx in indices:
                sizes[idx] += 1
        elif diff < 0:
            # Remove excess items from largest categories
            for _ in range(-diff):
                idx = np.argmax(sizes)
                sizes[idx] -= 1
                
        return sizes.tolist()
    
    def add_new_content(self, timestep: int) -> List[Item]:
        """
        Add new content items to the universe based on growth rate.
        
        This simulates the introduction of new content over time,
        which is a critical aspect of real-world recommendation systems.
        
        Args:
            timestep: Current simulation timestep
            
        Returns:
            List of newly added items
        """
        if not self.dynamic_content:
            return []
            
        # Determine how many items to add based on growth rate
        num_new_items = max(1, int(np.ceil(len(self.items) * self.content_growth_rate)))
        
        # Generate new items
        new_items = []
        
        # Generate popularity scores for new items (typically lower than existing content)
        base_popularity = np.min([item.popularity_score for item in self.items])
        new_popularity = self.rng.uniform(0.1 * base_popularity, base_popularity, num_new_items)
        
        for i in range(num_new_items):
            # Randomly select a primary category
            primary_category = self.rng.integers(0, self.num_categories)
            
            # Get the category center by averaging existing items
            if self.category_items[primary_category]:
                category_items_features = np.array([item.features for item in self.category_items[primary_category]])
                category_center = np.mean(category_items_features, axis=0)
            else:
                # If no items in category, create random center
                category_center = self.rng.normal(0, 1, self.num_features)
                category_center = category_center / np.linalg.norm(category_center)
            
            # Generate item features as category center + noise
            features = category_center + self.rng.normal(0, CONTENT_FEATURE_NOISE, self.num_features)

            # Normalize features
            features = features / np.linalg.norm(features)
            
            # Assign primary category + possible additional categories
            num_categories = 1 + self.rng.binomial(2, CONTENT_MULTI_CATEGORY_PROBABILITY)  # 1-3 categories per item

            categories = [primary_category]  # Primary category
            if num_categories > 1:
                # Add additional random categories
                possible_additional = [c for c in range(self.num_categories) if c != primary_category]
                if possible_additional:
                    additional_categories = self.rng.choice(
                        possible_additional,
                        size=min(num_categories-1, len(possible_additional)),
                        replace=False
                    )
                    categories.extend(additional_categories)
            
            item = Item(
                item_id=self.next_item_id,
                features=features,
                categories=categories,
                popularity_score=new_popularity[i],
                creation_time=timestep
            )
            
            new_items.append(item)
            self.items.append(item)
            self.item_id_map[self.next_item_id] = item
            for cat in categories:
                self.category_items[cat].add(item)
            
            self.next_item_id += 1
            
        return new_items
    
    def get_items_by_category(self, category_id: int) -> List[Item]:
        """
        Retrieve all items belonging to a specific category.
        
        Args:
            category_id: The category identifier
            
        Returns:
            List of items in the specified category
        """
        return list(self.category_items.get(category_id, set()))
    
    def get_item_by_id(self, item_id: int) -> Optional[Item]:
        """
        Retrieve a specific item by its ID.
        
        Args:
            item_id: The item identifier
            
        Returns:
            The requested item or None if not found
        """
        return self.item_id_map.get(item_id)
    
    def get_similar_items(self, item: Item, n: int = 10, exclude_ids: Optional[Set[int]] = None) -> List[Item]:
        """
        Find the n most similar items to a given item based on feature similarity.
        
        Args:
            item: The reference item
            n: Number of similar items to return
            exclude_ids: Set of item IDs to exclude from results
            
        Returns:
            List of n most similar items
        """
        if exclude_ids is None:
            exclude_ids = set()
            
        # First try items from the same categories for efficiency
        category_items = set()
        for cat in item.categories:
            category_items.update([
                i for i in self.category_items[cat]
                if i.item_id != item.item_id and i.item_id not in exclude_ids
            ])
        
        # If not enough items in shared categories, include other items
        if len(category_items) < n:
            other_items = [
                i for i in self.items
                if i.item_id != item.item_id and i.item_id not in exclude_ids
                and i not in category_items
            ]
            category_items.update(other_items)
        
        # Calculate similarities
        category_items = list(category_items)
        similarities = [(i, item.similarity(i)) for i in category_items]
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n
        return [i for i, _ in similarities[:n]]
    
    def get_popularity_distribution(self) -> np.ndarray:
        """
        Get the popularity distribution across all items.
        
        Returns:
            Array of popularity scores for each item
        """
        return np.array([item.popularity_score for item in self.items])
    
    def get_new_content_since(self, timestep: int) -> List[Item]:
        """
        Get content items added since a specific timestep.
        
        Args:
            timestep: Reference timestep
            
        Returns:
            List of items added after the reference timestep
        """
        return [item for item in self.items if item.creation_time > timestep]
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the content universe to a pandas DataFrame for analysis.
        
        Returns:
            DataFrame with item features, categories, and metadata
        """
        data = {
            'item_id': [],
            'primary_category': [],
            'categories': [],
            'num_categories': [],
            'popularity_score': [],
            'creation_time': []
        }
        
        # Add feature columns
        for i in range(self.num_features):
            data[f'feature_{i}'] = []
            
        # Add metadata columns if any
        metadata_keys = set()
        for item in self.items:
            metadata_keys.update(item.metadata.keys())
            
        for key in metadata_keys:
            data[f'metadata_{key}'] = []
            
        # Populate data
        for item in self.items:
            data['item_id'].append(item.item_id)
            data['primary_category'].append(item.categories[0] if item.categories else -1)
            data['categories'].append(item.categories)
            data['num_categories'].append(len(item.categories))
            data['popularity_score'].append(item.popularity_score)
            data['creation_time'].append(item.creation_time)
            
            for i, value in enumerate(item.features):
                data[f'feature_{i}'].append(value)
                
            for key in metadata_keys:
                data[f'metadata_{key}'].append(item.metadata.get(key, None))
                
        return pd.DataFrame(data)