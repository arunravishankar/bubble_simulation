import numpy as np
import pandas as pd
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..content.content_universe import ContentUniverse
from ..users.user_universe import UserUniverse
from ..recommenders.base import BaseRecommender
from ..settings import (
    SIM_NUM_TIMESTEPS, SIM_RECOMMENDATIONS_PER_STEP,
)

logger = logging.getLogger(__name__)

@dataclass
class SimulationResults:
    """
    Container for simulation results and metrics.
    """
    # Basic simulation info
    recommender_name: str
    num_timesteps: int
    num_users: int
    num_items: int
    
    # Time series data (one entry per timestep)
    timesteps: List[int] = field(default_factory=list)
    total_interactions: List[int] = field(default_factory=list)
    total_engagements: List[int] = field(default_factory=list)
    engagement_rate: List[float] = field(default_factory=list)
    
    # Diversity metrics over time
    content_diversity: List[float] = field(default_factory=list)  # Unique items shown / total items
    category_diversity: List[float] = field(default_factory=list)  # Category distribution entropy
    user_content_diversity: List[float] = field(default_factory=list)  # Avg diversity per user
    
    # Business metrics over time
    average_ctr: List[float] = field(default_factory=list)
    catalog_coverage: List[float] = field(default_factory=list)  # % of catalog shown
    
    # User-level metrics (aggregated per timestep)
    avg_user_engagement_rate: List[float] = field(default_factory=list)
    avg_user_diversity: List[float] = field(default_factory=list)
    
    # Recommender-specific metrics
    recommender_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    # Raw interaction data
    interaction_history: List[Dict] = field(default_factory=list)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert time series metrics to a pandas DataFrame."""
        data = {
            'timestep': self.timesteps,
            'total_interactions': self.total_interactions,
            'total_engagements': self.total_engagements,
            'engagement_rate': self.engagement_rate,
            'content_diversity': self.content_diversity,
            'category_diversity': self.category_diversity,
            'user_content_diversity': self.user_content_diversity,
            'average_ctr': self.average_ctr,
            'catalog_coverage': self.catalog_coverage,
            'avg_user_engagement_rate': self.avg_user_engagement_rate,
            'avg_user_diversity': self.avg_user_diversity
        }
        
        # Add recommender-specific metrics
        for metric_name, values in self.recommender_metrics.items():
            data[f'recommender_{metric_name}'] = values
            
        return pd.DataFrame(data)


class Simulator:
    """
    Main simulation engine that orchestrates the recommendation system simulation.
    
    This class manages the interaction between users, content, and recommenders
    over multiple timesteps to demonstrate filter bubble formation and prevention.
    """
    
    def __init__(self,
                 content_universe: ContentUniverse,
                 user_universe: UserUniverse,
                 recommender: BaseRecommender,
                 num_timesteps: int = SIM_NUM_TIMESTEPS,
                 recommendations_per_step: int = SIM_RECOMMENDATIONS_PER_STEP,
                 seed: Optional[int] = None):
        """
        Initialize the simulator.
        
        Args:
            content_universe: The content catalog
            user_universe: The user population
            recommender: The recommendation algorithm to test
            num_timesteps: Number of simulation timesteps to run
            recommendations_per_step: Number of recommendations per user per timestep
            seed: Random seed for reproducibility
        """
        self.content_universe = content_universe
        self.user_universe = user_universe
        self.recommender = recommender
        self.num_timesteps = num_timesteps
        self.recommendations_per_step = recommendations_per_step
        
        if seed is not None:
            np.random.seed(seed)
            
        # Track simulation state
        self.current_timestep = 0
        self.all_interactions = []  # Store all interactions across time
        self.timestep_interactions = []  # Current timestep interactions
        
        # Metrics tracking
        self.shown_items_global = set()  # All items shown across all users
        self.user_shown_items = {}  # Items shown to each user
        
        for user in self.user_universe.users:
            self.user_shown_items[user.user_id] = set()

    def run_simulation(self) -> SimulationResults:
        """
        Run the complete simulation.
        
        Returns:
            SimulationResults object containing all metrics and data
        """
        logger.info(f"Starting simulation with {self.recommender.name} for {self.num_timesteps} timesteps")
        start_time = time.time()
        
        # Initialize results container
        results = SimulationResults(
            recommender_name=self.recommender.name,
            num_timesteps=self.num_timesteps,
            num_users=len(self.user_universe.users),
            num_items=len(self.content_universe.items)
        )
        
        # Initialize user shown items tracking
        for user in self.user_universe.users:
            self.user_shown_items[user.user_id] = set()
        
        # Generate initial interactions if needed
        if not self.all_interactions:
            logger.info("Generating initial user interactions")
            initial_interactions = self.user_universe.generate_initial_interactions(
                self.content_universe
            )
            self.all_interactions.extend(initial_interactions)
        
        # Run simulation timesteps
        for timestep in range(self.num_timesteps):
            self.current_timestep = timestep
            logger.debug(f"Running timestep {timestep}")
            
            # Run one timestep
            timestep_metrics = self._run_timestep()
            
            # Record metrics
            self._record_metrics(results, timestep_metrics)
            
            # Add new content if dynamic
            if self.content_universe.dynamic_content:
                new_items = self.content_universe.add_new_content(timestep)
                if new_items:
                    logger.debug(f"Added {len(new_items)} new items at timestep {timestep}")
        
        # Calculate final metrics
        total_time = time.time() - start_time
        logger.info(f"Simulation completed in {total_time:.2f} seconds")
        
        return results
    
    def _run_timestep(self) -> Dict[str, Any]:
        """
        Run a single timestep of the simulation.
        
        Returns:
            Dictionary of metrics for this timestep
        """
        self.timestep_interactions = []
        timestep_engagements = 0
        
        # Update recommender with recent data
        if self.current_timestep == 0 and len(self.all_interactions) > 0:
            # Train on initial interactions
            self.recommender.train(
                self.all_interactions,
                self.content_universe.items,
                self.user_universe.users
            )
        elif self.current_timestep > 0:
            # Use all interactions for training (not just recent)
            self.recommender.update(
                self.all_interactions,  # Use all interactions, not just recent
                self.content_universe.items,
                self.user_universe.users,
                self.current_timestep
            )
        
        # Generate recommendations and interactions for each user
        for user in self.user_universe.users:
            user.start_new_session()
            
            # Get recommendations for this user
            try:
                recommendations = self.recommender.recommend(
                    user=user,
                    items=self.content_universe.items,
                    n=self.recommendations_per_step
                )
            except Exception as e:
                logger.warning(f"Recommendation failed for user {user.user_id}: {e}")
                recommendations = []
            
            # Process each recommendation
            for position, item in enumerate(recommendations[:user.attention_span]):
                # Record that this item was shown
                self.shown_items_global.add(item.item_id)
                self.user_shown_items[user.user_id].add(item.item_id)
                
                # Determine if user engages
                engaged = user.engage_with_item(
                    item=item,
                    position=position,
                    engagement_model=self.user_universe.engagement_model
                )
                
                # Record interaction
                interaction = {
                    'timestep': self.current_timestep,
                    'user_id': user.user_id,
                    'item_id': item.item_id,
                    'position': position,
                    'engaged': engaged,
                    'timestamp': self.current_timestep * 1000 + position  # Pseudo-timestamp
                }
                
                self.timestep_interactions.append(interaction)
                if engaged:
                    timestep_engagements += 1
        
        # Add timestep interactions to global history
        self.all_interactions.extend(self.timestep_interactions)
        
        # Calculate timestep metrics
        metrics = {
            'total_interactions': len(self.timestep_interactions),
            'total_engagements': timestep_engagements,
            'engagement_rate': timestep_engagements / max(1, len(self.timestep_interactions)),
            'unique_items_shown': len(set(interaction['item_id'] for interaction in self.timestep_interactions)),
            'unique_users_active': len(set(interaction['user_id'] for interaction in self.timestep_interactions))
        }
        
        return metrics
    
    def _record_metrics(self, results: SimulationResults, timestep_metrics: Dict[str, Any]) -> None:
        """
        Record metrics for the current timestep.
        
        Args:
            results: Results container to update
            timestep_metrics: Metrics from current timestep
        """
        # Basic metrics
        results.timesteps.append(self.current_timestep)
        results.total_interactions.append(timestep_metrics['total_interactions'])
        results.total_engagements.append(timestep_metrics['total_engagements'])
        results.engagement_rate.append(timestep_metrics['engagement_rate'])
        
        # Diversity metrics
        content_diversity = self._calculate_content_diversity()
        category_diversity = self._calculate_category_diversity()
        user_content_diversity = self._calculate_user_content_diversity()
        
        results.content_diversity.append(content_diversity)
        results.category_diversity.append(category_diversity)
        results.user_content_diversity.append(user_content_diversity)
        
        # Business metrics
        catalog_coverage = len(self.shown_items_global) / len(self.content_universe.items)
        results.catalog_coverage.append(catalog_coverage)
        
        # Calculate average CTR (same as engagement rate for this simulation)
        results.average_ctr.append(timestep_metrics['engagement_rate'])
        
        # User-level aggregated metrics
        user_engagement_rates = []
        user_diversities = []
        
        for user in self.user_universe.users:
            user_engagement_rates.append(user.get_engagement_rate())
            user_diversities.append(user.get_diversity_metric())
            
        results.avg_user_engagement_rate.append(np.mean(user_engagement_rates))
        results.avg_user_diversity.append(np.mean(user_diversities))
        
        # Recommender-specific metrics
        recommender_metrics = self.recommender.get_metrics()
        for metric_name, values in recommender_metrics.items():
            if metric_name not in results.recommender_metrics:
                # Initialize with zeros for previous timesteps
                results.recommender_metrics[metric_name] = [0.0] * self.current_timestep
            
            # Add latest value (or 0 if no new value)
            if values:
                results.recommender_metrics[metric_name].append(values[-1])
            else:
                results.recommender_metrics[metric_name].append(0.0)
        
        # Ensure all recommender metrics have the same length
        for metric_name in results.recommender_metrics:
            while len(results.recommender_metrics[metric_name]) < len(results.timesteps):
                results.recommender_metrics[metric_name].append(0.0)
    
    def _calculate_content_diversity(self) -> float:
        """
        Calculate overall content diversity as proportion of catalog shown.
        
        Returns:
            Diversity score between 0 and 1
        """
        if not self.content_universe.items:
            return 0.0
            
        return len(self.shown_items_global) / len(self.content_universe.items)
    
    def _calculate_category_diversity(self) -> float:
        """
        Calculate category diversity using entropy.
        
        Returns:
            Normalized entropy score between 0 and 1
        """
        if not self.timestep_interactions:
            return 0.0
            
        # Count interactions per category
        category_counts = {}
        for interaction in self.timestep_interactions:
            item = self.content_universe.get_item_by_id(interaction['item_id'])
            if item:
                for category in item.categories:
                    category_counts[category] = category_counts.get(category, 0) + 1
        
        if not category_counts:
            return 0.0
            
        # Calculate entropy
        total_interactions = sum(category_counts.values())
        entropy = 0.0
        
        for count in category_counts.values():
            if count > 0:
                prob = count / total_interactions
                entropy -= prob * np.log2(prob)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(category_counts))
        if max_entropy == 0:
            return 0.0
            
        return entropy / max_entropy
    
    def _calculate_user_content_diversity(self) -> float:
        """
        Calculate average content diversity across all users.
        
        Returns:
            Average diversity score between 0 and 1
        """
        diversities = []
        
        for user_id, shown_items in self.user_shown_items.items():
            if shown_items:
                # Diversity is proportion of catalog this user has seen
                user_diversity = len(shown_items) / len(self.content_universe.items)
                diversities.append(user_diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def get_interaction_dataframe(self) -> pd.DataFrame:
        """
        Get all interactions as a pandas DataFrame.
        
        Returns:
            DataFrame with all interaction data
        """
        if not self.all_interactions:
            return pd.DataFrame()
        
        # Ensure all interactions have required fields
        cleaned_interactions = []
        for interaction in self.all_interactions:
            cleaned_interaction = interaction.copy()
            # Ensure timestep exists
            if 'timestep' not in cleaned_interaction:
                cleaned_interaction['timestep'] = 0  # Default to 0 for initial interactions
            cleaned_interactions.append(cleaned_interaction)
            
        df = pd.DataFrame(cleaned_interactions)
        
        # Convert timestep to int explicitly
        df['timestep'] = df['timestep'].fillna(0).astype(int)
        
        # Add item and user features
        item_features = []
        user_features = []
        
        for _, row in df.iterrows():
            # Get item info
            item = self.content_universe.get_item_by_id(row['item_id'])
            if item:
                item_features.append({
                    'item_categories': item.categories,
                    'item_popularity': item.popularity_score,
                    'item_creation_time': item.creation_time
                })
            else:
                item_features.append({
                    'item_categories': [],
                    'item_popularity': 0.0,
                    'item_creation_time': 0
                })
            
            # Get user info
            user = self.user_universe.get_user_by_id(row['user_id'])
            if user:
                user_features.append({
                    'user_exploration_factor': user.exploration_factor,
                    'user_position_bias_factor': user.position_bias_factor,
                    'user_diversity_preference': user.diversity_preference
                })
            else:
                user_features.append({
                    'user_exploration_factor': 0.0,
                    'user_position_bias_factor': 0.0,
                    'user_diversity_preference': 0.0
                })
        
        # Add features to dataframe using assignment instead of concatenation
        item_df = pd.DataFrame(item_features)
        user_df = pd.DataFrame(user_features)
        
        # Assign columns individually to avoid alignment issues
        for col in item_df.columns:
            df[col] = item_df[col].values
        
        for col in user_df.columns:
            df[col] = user_df[col].values
        
        return df
    
    def reset_simulation(self) -> None:
        """Reset simulation state for a new run."""
        self.current_timestep = 0
        self.all_interactions = []
        self.timestep_interactions = []
        self.shown_items_global = set()
        self.user_shown_items = {user.user_id: set() for user in self.user_universe.users}
        
        # Reset user states
        for user in self.user_universe.users:
            user.interaction_history = []
            user.category_interests = {}
            user.engagement_counts = {
                'total_recommended': 0,
                'total_engaged': 0,
                'last_session_recommended': 0,
                'last_session_engaged': 0
            }


def run_comparative_simulation(content_universe: ContentUniverse,
                              user_universe: UserUniverse,
                              recommenders: List[BaseRecommender],
                              num_timesteps: int = SIM_NUM_TIMESTEPS,
                              recommendations_per_step: int = SIM_RECOMMENDATIONS_PER_STEP,
                              seed: Optional[int] = None) -> Dict[str, SimulationResults]:
    """
    Run comparative simulations with multiple recommenders.
    
    Args:
        content_universe: The content catalog
        user_universe: The user population  
        recommenders: List of recommenders to compare
        num_timesteps: Number of timesteps for each simulation
        recommendations_per_step: Recommendations per user per timestep
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping recommender names to their results
    """
    results = {}
    
    for recommender in recommenders:
        logger.info(f"Running simulation for {recommender.name}")
        
        # Create fresh simulator for each recommender
        simulator = Simulator(
            content_universe=content_universe,
            user_universe=user_universe,
            recommender=recommender,
            num_timesteps=num_timesteps,
            recommendations_per_step=recommendations_per_step,
            seed=seed
        )
        
        # Run simulation
        sim_results = simulator.run_simulation()
        results[recommender.name] = sim_results
        
        # Reset for next simulation
        simulator.reset_simulation()
        
        # Also reset recommender state if possible
        if hasattr(recommender, 'reset_exploration'):
            recommender.reset_exploration()
    
    return results