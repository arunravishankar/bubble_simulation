import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from scipy import stats
from collections import Counter
import logging

from .simulator import SimulationResults
# from ..content.item import Item
# from ..users.user import User

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Comprehensive metrics calculator for recommendation system analysis.
    
    This class provides methods to calculate various metrics that demonstrate
    filter bubble formation, user engagement patterns, business outcomes,
    and the effectiveness of different recommendation strategies.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def calculate_diversity_metrics(self, 
                                  results: SimulationResults,
                                  interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive diversity metrics.
        
        Args:
            results: Simulation results object
            interactions_df: DataFrame with all interactions
            
        Returns:
            Dictionary of diversity metrics
        """
        metrics = {}
        
        # Content diversity over time
        metrics['content_diversity_trend'] = self._calculate_diversity_trend(results.content_diversity)
        metrics['content_diversity_final'] = results.content_diversity[-1] if results.content_diversity else 0.0
        metrics['content_diversity_change'] = self._calculate_metric_change(results.content_diversity)
        
        # Category diversity over time
        metrics['category_diversity_trend'] = self._calculate_diversity_trend(results.category_diversity)
        metrics['category_diversity_final'] = results.category_diversity[-1] if results.category_diversity else 0.0
        metrics['category_diversity_change'] = self._calculate_metric_change(results.category_diversity)
        
        # User-level diversity
        metrics['user_diversity_trend'] = self._calculate_diversity_trend(results.user_content_diversity)
        metrics['user_diversity_final'] = results.user_content_diversity[-1] if results.user_content_diversity else 0.0
        metrics['user_diversity_change'] = self._calculate_metric_change(results.user_content_diversity)
        
        # Bubble formation indicators
        metrics['bubble_formation_rate'] = self._calculate_bubble_formation_rate(results.content_diversity)
        metrics['diversity_gini_coefficient'] = self._calculate_diversity_gini(interactions_df)
        
        # Long-tail content exposure
        metrics['long_tail_exposure'] = self._calculate_long_tail_exposure(interactions_df)
        
        return metrics
    
    def calculate_engagement_metrics(self, 
                                   results: SimulationResults,
                                   interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate engagement and user satisfaction metrics.
        
        Args:
            results: Simulation results object
            interactions_df: DataFrame with all interactions
            
        Returns:
            Dictionary of engagement metrics
        """
        metrics = {}
        
        # Overall engagement trends
        metrics['engagement_rate_trend'] = self._calculate_diversity_trend(results.engagement_rate)
        metrics['engagement_rate_final'] = results.engagement_rate[-1] if results.engagement_rate else 0.0
        metrics['engagement_rate_change'] = self._calculate_metric_change(results.engagement_rate)
        
        # User-level engagement patterns
        metrics['user_engagement_trend'] = self._calculate_diversity_trend(results.avg_user_engagement_rate)
        metrics['user_engagement_variance'] = self._calculate_engagement_variance(interactions_df)
        
        # Position bias effects
        metrics['position_bias_impact'] = self._calculate_position_bias_impact(interactions_df)
        
        # Engagement by content characteristics
        metrics['engagement_by_novelty'] = self._calculate_engagement_by_novelty(interactions_df)
        metrics['engagement_by_popularity'] = self._calculate_engagement_by_popularity(interactions_df)
        
        # User satisfaction proxies
        metrics['session_length_trend'] = self._calculate_session_length_trends(interactions_df)
        metrics['repeat_engagement_rate'] = self._calculate_repeat_engagement(interactions_df)
        
        return metrics
    
    def calculate_business_metrics(self, 
                                 results: SimulationResults,
                                 interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate business and revenue-related metrics.
        
        Args:
            results: Simulation results object
            interactions_df: DataFrame with all interactions
            
        Returns:
            Dictionary of business metrics
        """
        metrics = {}
        
        # Catalog utilization
        metrics['catalog_coverage_trend'] = self._calculate_diversity_trend(results.catalog_coverage)
        metrics['catalog_coverage_final'] = results.catalog_coverage[-1] if results.catalog_coverage else 0.0
        metrics['catalog_utilization_efficiency'] = self._calculate_catalog_efficiency(interactions_df)
        
        # Revenue proxies (based on engagement and content characteristics)
        metrics['estimated_revenue_trend'] = self._calculate_revenue_proxy(interactions_df, results.timesteps)
        metrics['revenue_per_user'] = self._calculate_revenue_per_user(interactions_df)
        
        # User retention indicators
        metrics['user_retention_rate'] = self._calculate_user_retention(interactions_df)
        metrics['user_churn_indicators'] = self._calculate_churn_indicators(interactions_df)
        
        # Content inventory efficiency
        metrics['content_turnover_rate'] = self._calculate_content_turnover(interactions_df)
        metrics['niche_content_performance'] = self._calculate_niche_performance(interactions_df)
        
        return metrics
    
    def calculate_bubble_formation_metrics(self, 
                                         results: SimulationResults,
                                         interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate specific metrics related to filter bubble formation.
        
        Args:
            results: Simulation results object
            interactions_df: DataFrame with all interactions
            
        Returns:
            Dictionary of bubble formation metrics
        """
        metrics = {}
        
        # Bubble formation rate (how quickly diversity decreases)
        metrics['bubble_formation_velocity'] = self._calculate_bubble_velocity(results.content_diversity)
        
        # Content concentration over time
        metrics['content_concentration_trend'] = self._calculate_concentration_trend(interactions_df)
        
        # User interest narrowing
        metrics['interest_narrowing_rate'] = self._calculate_interest_narrowing(interactions_df)
        
        # Serendipity loss (unexpected discoveries over time)
        metrics['serendipity_trend'] = self._calculate_serendipity_trend(interactions_df)
        
        # Echo chamber strength
        metrics['echo_chamber_strength'] = self._calculate_echo_chamber_strength(interactions_df)
        
        # Content novelty decay
        metrics['novelty_decay_rate'] = self._calculate_novelty_decay(interactions_df)
        
        return metrics
    
    def calculate_comparative_metrics(self, 
                                    results_dict: Dict[str, SimulationResults],
                                    interactions_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate comparative metrics across different recommender systems.
        
        Args:
            results_dict: Dictionary mapping recommender names to results
            interactions_dict: Dictionary mapping recommender names to interaction DataFrames
            
        Returns:
            Dictionary of comparative metrics
        """
        metrics = {}
        
        # Diversity comparison
        diversity_comparison = {}
        for name, results in results_dict.items():
            diversity_comparison[name] = {
                'final_diversity': results.content_diversity[-1] if results.content_diversity else 0.0,
                'diversity_change': self._calculate_metric_change(results.content_diversity),
                'diversity_stability': self._calculate_stability(results.content_diversity)
            }
        metrics['diversity_comparison'] = diversity_comparison
        
        # Engagement comparison
        engagement_comparison = {}
        for name, results in results_dict.items():
            engagement_comparison[name] = {
                'final_engagement': results.engagement_rate[-1] if results.engagement_rate else 0.0,
                'engagement_change': self._calculate_metric_change(results.engagement_rate),
                'engagement_stability': self._calculate_stability(results.engagement_rate)
            }
        metrics['engagement_comparison'] = engagement_comparison
        
        # Business impact comparison
        business_comparison = {}
        for name, interactions_df in interactions_dict.items():
            business_comparison[name] = {
                'catalog_utilization': len(interactions_df['item_id'].unique()) / interactions_df['item_id'].nunique() if len(interactions_df) > 0 else 0.0,
                'user_retention': self._calculate_user_retention(interactions_df),
                'revenue_proxy': self._calculate_total_revenue_proxy(interactions_df)
            }
        metrics['business_comparison'] = business_comparison
        
        # Statistical significance tests
        metrics['statistical_tests'] = self._perform_significance_tests(results_dict, interactions_dict)
        
        return metrics
    
    # Helper methods for metric calculations
    
    def _calculate_diversity_trend(self, values: List[float]) -> str:
        """Calculate whether diversity is increasing, decreasing, or stable."""
        if len(values) < 2:
            return "insufficient_data"
        
        slope, _, _, p_value, _ = stats.linregress(range(len(values)), values)
        
        if p_value > 0.05:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _calculate_metric_change(self, values: List[float]) -> float:
        """Calculate the percentage change from first to last value."""
        if len(values) < 2:
            return 0.0
        
        start_val = values[0] if values[0] != 0 else 0.001  # Avoid division by zero
        end_val = values[-1]
        
        return (end_val - start_val) / start_val * 100
    
    def _calculate_bubble_formation_rate(self, diversity_values: List[float]) -> float:
        """Calculate how quickly filter bubbles form (diversity decrease rate)."""
        if len(diversity_values) < 2:
            return 0.0
        
        # Calculate the slope of diversity decline
        slope, _, _, _, _ = stats.linregress(range(len(diversity_values)), diversity_values)
        
        # Return absolute slope (positive value indicates bubble formation)
        return abs(slope) if slope < 0 else 0.0
    
    def _calculate_diversity_gini(self, interactions_df: pd.DataFrame) -> float:
        """Calculate Gini coefficient for content distribution."""
        if len(interactions_df) == 0:
            return 0.0
        
        # Count interactions per item
        item_counts = interactions_df['item_id'].value_counts().values
        
        # Calculate Gini coefficient
        n = len(item_counts)
        if n == 0:
            return 0.0
        
        # Sort values
        sorted_counts = np.sort(item_counts)
        
        # Calculate Gini
        cumsum = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return max(0.0, gini)  # Ensure non-negative
    
    def _calculate_long_tail_exposure(self, interactions_df: pd.DataFrame) -> float:
        """Calculate what fraction of interactions are with long-tail (less popular) content."""
        if len(interactions_df) == 0:
            return 0.0
        
        # Count interactions per item
        item_counts = interactions_df['item_id'].value_counts()
        
        # Define long-tail as bottom 80% of items by interaction count
        total_items = len(item_counts)
        long_tail_threshold = int(total_items * 0.8)
        
        long_tail_items = set(item_counts.iloc[long_tail_threshold:].index)
        long_tail_interactions = interactions_df[interactions_df['item_id'].isin(long_tail_items)]
        
        return len(long_tail_interactions) / len(interactions_df)
    
    def _calculate_engagement_variance(self, interactions_df: pd.DataFrame) -> float:
        """Calculate variance in engagement rates across users."""
        if len(interactions_df) == 0:
            return 0.0
        
        user_engagement = interactions_df.groupby('user_id')['engaged'].mean()
        return float(user_engagement.var())
    
    def _calculate_position_bias_impact(self, interactions_df: pd.DataFrame) -> float:
        """Calculate the impact of position on engagement rates."""
        if len(interactions_df) == 0 or 'position' not in interactions_df.columns:
            return 0.0
        
        position_engagement = interactions_df.groupby('position')['engaged'].mean()
        
        if len(position_engagement) < 2:
            return 0.0
        
        # Calculate correlation between position and engagement
        positions = position_engagement.index.values
        engagement_rates = position_engagement.values
        
        correlation, _ = stats.pearsonr(positions, engagement_rates)
        return abs(correlation)  # Return absolute correlation
    
    def _calculate_engagement_by_novelty(self, interactions_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate engagement rates by content novelty."""
        if len(interactions_df) == 0:
            return {'high_novelty': 0.0, 'low_novelty': 0.0}
        
        # Use timestamp as a proxy for novelty (newer = more novel)
        if 'timestamp' in interactions_df.columns:
            median_timestamp = interactions_df['timestamp'].median()
            high_novelty = interactions_df[interactions_df['timestamp'] >= median_timestamp]
            low_novelty = interactions_df[interactions_df['timestamp'] < median_timestamp]
            
            return {
                'high_novelty': high_novelty['engaged'].mean() if len(high_novelty) > 0 else 0.0,
                'low_novelty': low_novelty['engaged'].mean() if len(low_novelty) > 0 else 0.0
            }
        
        return {'high_novelty': 0.0, 'low_novelty': 0.0}
    
    def _calculate_engagement_by_popularity(self, interactions_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate engagement rates by content popularity."""
        if len(interactions_df) == 0 or 'item_popularity' not in interactions_df.columns:
            return {'high_popularity': 0.0, 'low_popularity': 0.0}
        
        median_popularity = interactions_df['item_popularity'].median()
        high_pop = interactions_df[interactions_df['item_popularity'] >= median_popularity]
        low_pop = interactions_df[interactions_df['item_popularity'] < median_popularity]
        
        return {
            'high_popularity': high_pop['engaged'].mean() if len(high_pop) > 0 else 0.0,
            'low_popularity': low_pop['engaged'].mean() if len(low_pop) > 0 else 0.0
        }
    
    def _calculate_session_length_trends(self, interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trends in user session lengths over time."""
        if len(interactions_df) == 0:
            return {'trend': 'stable', 'average_length': 0.0}
        
        # Group by user and timestep to get session lengths
        session_lengths = interactions_df.groupby(['user_id', 'timestep']).size()
        timestep_avg_lengths = session_lengths.groupby('timestep').mean()
        
        if len(timestep_avg_lengths) < 2:
            return {'trend': 'stable', 'average_length': float(timestep_avg_lengths.mean()) if len(timestep_avg_lengths) > 0 else 0.0}
        
        trend = self._calculate_diversity_trend(timestep_avg_lengths.tolist())
        
        return {
            'trend': trend,
            'average_length': float(timestep_avg_lengths.mean()),
            'length_change': self._calculate_metric_change(timestep_avg_lengths.tolist())
        }
    
    def _calculate_repeat_engagement(self, interactions_df: pd.DataFrame) -> float:
        """Calculate the rate of repeat engagement with same content."""
        if len(interactions_df) == 0:
            return 0.0
        
        # Find users who engaged with the same item multiple times
        user_item_engagements = interactions_df[interactions_df['engaged']].groupby(['user_id', 'item_id']).size()
        repeat_engagements = user_item_engagements[user_item_engagements > 1]
        
        total_engaged = len(interactions_df[interactions_df['engaged']])
        return len(repeat_engagements) / max(1, total_engaged)
    
    def _calculate_catalog_efficiency(self, interactions_df: pd.DataFrame) -> float:
        """Calculate how efficiently the catalog is being utilized."""
        if len(interactions_df) == 0:
            return 0.0
        
        # Calculate the ratio of unique items shown to total interactions
        unique_items = interactions_df['item_id'].nunique()
        total_interactions = len(interactions_df)
        
        return unique_items / max(1, total_interactions)
    
    def _calculate_revenue_proxy(self, interactions_df: pd.DataFrame, timesteps: List[int]) -> List[float]:
        """Calculate a proxy for revenue over time based on engagements and popularity."""
        revenue_by_timestep = []
        
        for timestep in timesteps:
            timestep_data = interactions_df[interactions_df['timestep'] == timestep]
            
            if len(timestep_data) == 0:
                revenue_by_timestep.append(0.0)
                continue
            
            # Revenue proxy: engagements weighted by item popularity
            engaged_data = timestep_data[timestep_data['engaged']]
            
            if len(engaged_data) == 0 or 'item_popularity' not in engaged_data.columns:
                revenue_by_timestep.append(0.0)
            else:
                revenue = engaged_data['item_popularity'].sum()
                revenue_by_timestep.append(float(revenue))
        
        return revenue_by_timestep
    
    def _calculate_revenue_per_user(self, interactions_df: pd.DataFrame) -> float:
        """Calculate average revenue proxy per user."""
        if len(interactions_df) == 0:
            return 0.0
        
        engaged_data = interactions_df[interactions_df['engaged']]
        
        if len(engaged_data) == 0 or 'item_popularity' not in engaged_data.columns:
            return 0.0
        
        revenue_by_user = engaged_data.groupby('user_id')['item_popularity'].sum()
        return float(revenue_by_user.mean())
    
    def _calculate_user_retention(self, interactions_df: pd.DataFrame) -> float:
        """Calculate user retention rate across timesteps."""
        if len(interactions_df) == 0 or 'timestep' not in interactions_df.columns:
            return 0.0
        
        # Calculate what fraction of users remain active across timesteps
        user_timesteps = interactions_df.groupby('user_id')['timestep'].nunique()
        max_timesteps = interactions_df['timestep'].nunique()
        
        if max_timesteps <= 1:
            return 1.0
        
        # Users who appear in most timesteps are considered "retained"
        retention_threshold = max_timesteps * 0.7  # 70% of timesteps
        retained_users = user_timesteps[user_timesteps >= retention_threshold]
        
        return len(retained_users) / len(user_timesteps)
    
    def _calculate_churn_indicators(self, interactions_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate indicators of user churn."""
        if len(interactions_df) == 0:
            return {'early_churn_rate': 0.0, 'engagement_decline_rate': 0.0}
        
        # Early churn: users who stop interacting after first few timesteps
        user_last_timestep = interactions_df.groupby('user_id')['timestep'].max()
        max_timestep = interactions_df['timestep'].max()
        
        early_churn_threshold = max(1, max_timestep * 0.3)  # First 30% of simulation
        early_churned = user_last_timestep[user_last_timestep <= early_churn_threshold]
        early_churn_rate = len(early_churned) / len(user_last_timestep)
        
        # Engagement decline: users whose engagement rate decreases over time
        user_engagement_trends = []
        for user_id in interactions_df['user_id'].unique():
            user_data = interactions_df[interactions_df['user_id'] == user_id]
            if len(user_data) > 1:
                timestep_engagement = user_data.groupby('timestep')['engaged'].mean()
                if len(timestep_engagement) > 1:
                    slope, _, _, _, _ = stats.linregress(timestep_engagement.index, timestep_engagement.values)
                    user_engagement_trends.append(slope)
        
        declining_users = [slope for slope in user_engagement_trends if slope < -0.01]  # Significant decline
        engagement_decline_rate = len(declining_users) / max(1, len(user_engagement_trends))
        
        return {
            'early_churn_rate': early_churn_rate,
            'engagement_decline_rate': engagement_decline_rate
        }
    
    def _calculate_content_turnover(self, interactions_df: pd.DataFrame) -> float:
        """Calculate how quickly new content gets introduced and adopted."""
        if len(interactions_df) == 0:
            return 0.0
        
        # Calculate the rate at which new items appear in recommendations
        timestep_new_items = {}
        seen_items = set()
        
        for timestep in sorted(interactions_df['timestep'].unique()):
            timestep_data = interactions_df[interactions_df['timestep'] == timestep]
            timestep_items = set(timestep_data['item_id'].unique())
            new_items = timestep_items - seen_items
            timestep_new_items[timestep] = len(new_items)
            seen_items.update(timestep_items)
        
        # Return average new items per timestep
        return np.mean(list(timestep_new_items.values()))
    
    def _calculate_niche_performance(self, interactions_df: pd.DataFrame) -> float:
        """Calculate how well niche (low popularity) content performs."""
        if len(interactions_df) == 0 or 'item_popularity' not in interactions_df.columns:
            return 0.0
        
        # Define niche content as bottom 25% by popularity
        popularity_threshold = interactions_df['item_popularity'].quantile(0.25)
        niche_data = interactions_df[interactions_df['item_popularity'] <= popularity_threshold]
        
        if len(niche_data) == 0:
            return 0.0
        
        return niche_data['engaged'].mean()
    
    def _calculate_bubble_velocity(self, diversity_values: List[float]) -> float:
        """Calculate how quickly filter bubbles form."""
        if len(diversity_values) < 3:
            return 0.0
        
        # Calculate second derivative to measure acceleration of diversity loss
        first_derivative = np.diff(diversity_values)
        second_derivative = np.diff(first_derivative)
        
        # Return average acceleration (more negative = faster bubble formation)
        return abs(np.mean(second_derivative)) if len(second_derivative) > 0 else 0.0
    
    def _calculate_concentration_trend(self, interactions_df: pd.DataFrame) -> List[float]:
        """Calculate content concentration (opposite of diversity) over time."""
        concentration_by_timestep = []
        
        for timestep in sorted(interactions_df['timestep'].unique()):
            timestep_data = interactions_df[interactions_df['timestep'] == timestep]
            
            if len(timestep_data) == 0:
                concentration_by_timestep.append(0.0)
                continue
            
            # Calculate Herfindahl index (measure of concentration)
            item_counts = timestep_data['item_id'].value_counts()
            total_interactions = len(timestep_data)
            
            herfindahl = sum((count / total_interactions) ** 2 for count in item_counts)
            concentration_by_timestep.append(herfindahl)
        
        return concentration_by_timestep
    
    def _calculate_interest_narrowing(self, interactions_df: pd.DataFrame) -> float:
        """Calculate how much user interests narrow over time."""
        if len(interactions_df) == 0 or 'item_categories' not in interactions_df.columns:
            return 0.0
        
        # Calculate category diversity for each user over time
        user_narrowing_rates = []
        
        for user_id in interactions_df['user_id'].unique():
            user_data = interactions_df[interactions_df['user_id'] == user_id]
            
            if len(user_data) < 2:
                continue
            
            # Calculate unique categories per timestep for this user
            timestep_categories = []
            for timestep in sorted(user_data['timestep'].unique()):
                timestep_data = user_data[user_data['timestep'] == timestep]
                categories = set()
                for cats in timestep_data['item_categories']:
                    if isinstance(cats, list):
                        categories.update(cats)
                timestep_categories.append(len(categories))
            
            if len(timestep_categories) > 1:
                # Calculate trend in category diversity
                slope, _, _, _, _ = stats.linregress(range(len(timestep_categories)), timestep_categories)
                user_narrowing_rates.append(-slope)  # Negative slope = narrowing
        
        return np.mean(user_narrowing_rates) if user_narrowing_rates else 0.0
    
    def _calculate_serendipity_trend(self, interactions_df: pd.DataFrame) -> List[float]:
        """Calculate serendipity (unexpected discoveries) over time."""
        serendipity_by_timestep = []
        
        for timestep in sorted(interactions_df['timestep'].unique()):
            timestep_data = interactions_df[interactions_df['timestep'] == timestep]
            
            if len(timestep_data) == 0:
                serendipity_by_timestep.append(0.0)
                continue
            
            # Serendipity proxy: engagement with low-popularity items
            if 'item_popularity' in timestep_data.columns:
                low_pop_threshold = timestep_data['item_popularity'].quantile(0.3)
                low_pop_engaged = timestep_data[
                    (timestep_data['item_popularity'] <= low_pop_threshold) & 
                    (timestep_data['engaged'])
                ]
                serendipity = len(low_pop_engaged) / max(1, len(timestep_data))
            else:
                serendipity = 0.0
            
            serendipity_by_timestep.append(serendipity)
        
        return serendipity_by_timestep
    
    def _calculate_echo_chamber_strength(self, interactions_df: pd.DataFrame) -> float:
        """Calculate the strength of echo chambers (repeated similar content)."""
        if len(interactions_df) == 0 or 'item_categories' not in interactions_df.columns:
            return 0.0
        
        # Calculate category repetition for each user
        user_echo_scores = []
        
        for user_id in interactions_df['user_id'].unique():
            user_data = interactions_df[interactions_df['user_id'] == user_id]
            
            # Count category frequencies
            category_counts = Counter()
            for cats in user_data['item_categories']:
                if isinstance(cats, list):
                    category_counts.update(cats)
            
            if len(category_counts) == 0:
                continue
            
            # Calculate concentration (higher = more echo chamber-like)
            total_interactions = sum(category_counts.values())
            echo_score = sum((count / total_interactions) ** 2 for count in category_counts.values())
            user_echo_scores.append(echo_score)
        
        return np.mean(user_echo_scores) if user_echo_scores else 0.0
    
    def _calculate_novelty_decay(self, interactions_df: pd.DataFrame) -> float:
        """Calculate how quickly content novelty decays over time."""
        if len(interactions_df) == 0 or 'timestamp' not in interactions_df.columns:
            return 0.0
        
        # Calculate average "age" of content at each timestep
        timestep_novelty = []
        
        for timestep in sorted(interactions_df['timestep'].unique()):
            timestep_data = interactions_df[interactions_df['timestep'] == timestep]
            
            if len(timestep_data) == 0:
                continue
            
            # Novelty = how recent the content is relative to current timestep
            content_ages = timestep - timestep_data['timestamp']
            avg_age = content_ages.mean()
            novelty = 1.0 / (1.0 + avg_age)  # Inverse relationship
            timestep_novelty.append(novelty)
        
        if len(timestep_novelty) < 2:
            return 0.0
        
        # Calculate how fast novelty decays
        slope, _, _, _, _ = stats.linregress(range(len(timestep_novelty)), timestep_novelty)
        return abs(slope)  # Positive value indicates decay
    
    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability (inverse of variance) of a metric over time."""
        if len(values) < 2:
            return 1.0
        
        variance = np.var(values)
        return 1.0 / (1.0 + variance)  # Higher values = more stable
    
    def _calculate_total_revenue_proxy(self, interactions_df: pd.DataFrame) -> float:
        """Calculate total revenue proxy for the entire simulation."""
        if len(interactions_df) == 0:
            return 0.0
        
        engaged_data = interactions_df[interactions_df['engaged']]
        
        if len(engaged_data) == 0 or 'item_popularity' not in engaged_data.columns:
            return 0.0
        
        return float(engaged_data['item_popularity'].sum())
    
    def _perform_significance_tests(self, 
                                  results_dict: Dict[str, SimulationResults],
                                  interactions_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform statistical significance tests between recommenders."""
        significance_tests = {}
        
        recommender_names = list(results_dict.keys())
        
        if len(recommender_names) < 2:
            return significance_tests
        
        # Test diversity differences
        diversity_tests = {}
        for i, name1 in enumerate(recommender_names):
            for name2 in recommender_names[i+1:]:
                diversity1 = results_dict[name1].content_diversity
                diversity2 = results_dict[name2].content_diversity
                
                if len(diversity1) > 1 and len(diversity2) > 1:
                    statistic, p_value = stats.ttest_ind(diversity1, diversity2)
                    diversity_tests[f"{name1}_vs_{name2}"] = {
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
        
        significance_tests['diversity_tests'] = diversity_tests
        
        # Test engagement differences
        engagement_tests = {}
        for i, name1 in enumerate(recommender_names):
            for name2 in recommender_names[i+1:]:
                engagement1 = results_dict[name1].engagement_rate
                engagement2 = results_dict[name2].engagement_rate
                
                if len(engagement1) > 1 and len(engagement2) > 1:
                    statistic, p_value = stats.ttest_ind(engagement1, engagement2)
                    engagement_tests[f"{name1}_vs_{name2}"] = {
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05)
                    }
        
        significance_tests['engagement_tests'] = engagement_tests
        
        return significance_tests


class FilterBubbleAnalyzer:
    """
    Specialized analyzer for filter bubble detection and analysis.
    
    This class provides specific methods to detect, measure, and analyze
    filter bubble formation in recommendation systems.
    """
    
    def __init__(self):
        """Initialize the filter bubble analyzer."""
        self.metrics_calculator = MetricsCalculator()
    
    def analyze_bubble_formation(self, 
                                results: SimulationResults,
                                interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive analysis of filter bubble formation.
        
        Args:
            results: Simulation results object
            interactions_df: DataFrame with all interactions
            
        Returns:
            Dictionary containing bubble formation analysis
        """
        analysis = {}
        
        # Basic bubble metrics
        analysis['bubble_metrics'] = self.metrics_calculator.calculate_bubble_formation_metrics(
            results, interactions_df
        )
        
        # Diversity timeline analysis
        analysis['diversity_timeline'] = self._analyze_diversity_timeline(results)
        
        # User segmentation by bubble susceptibility
        analysis['user_segmentation'] = self._analyze_user_bubble_susceptibility(interactions_df)
        
        # Content type impact on bubbles
        analysis['content_impact'] = self._analyze_content_type_impact(interactions_df)
        
        # Bubble formation phases
        analysis['bubble_phases'] = self._identify_bubble_phases(results)
        
        return analysis
    
    def compare_bubble_prevention(self, 
                                results_dict: Dict[str, SimulationResults],
                                interactions_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Compare bubble prevention effectiveness across recommenders.
        
        Args:
            results_dict: Dictionary mapping recommender names to results
            interactions_dict: Dictionary mapping recommender names to interaction DataFrames
            
        Returns:
            Dictionary containing bubble prevention comparison
        """
        comparison = {}
        
        # Bubble formation comparison
        bubble_scores = {}
        for name, results in results_dict.items():
            bubble_scores[name] = self._calculate_bubble_score(results)
        
        comparison['bubble_scores'] = bubble_scores
        comparison['best_bubble_prevention'] = min(bubble_scores.items(), key=lambda x: x[1])
        comparison['worst_bubble_formation'] = max(bubble_scores.items(), key=lambda x: x[1])
        
        # Diversity preservation comparison
        diversity_preservation = {}
        for name, results in results_dict.items():
            diversity_preservation[name] = self._calculate_diversity_preservation(results)
        
        comparison['diversity_preservation'] = diversity_preservation
        
        # Trade-off analysis (diversity vs engagement)
        comparison['tradeoff_analysis'] = self._analyze_diversity_engagement_tradeoff(
            results_dict, interactions_dict
        )
        
        return comparison
    
    def generate_bubble_report(self, 
                             results_dict: Dict[str, SimulationResults],
                             interactions_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate a comprehensive filter bubble analysis report.
        
        Args:
            results_dict: Dictionary mapping recommender names to results
            interactions_dict: Dictionary mapping recommender names to interaction DataFrames
            
        Returns:
            Comprehensive report dictionary
        """
        report = {
            'executive_summary': {},
            'detailed_analysis': {},
            'recommendations': {},
            'statistical_evidence': {}
        }
        
        # Executive summary
        report['executive_summary'] = self._generate_executive_summary(
            results_dict, interactions_dict
        )
        
        # Detailed analysis for each recommender
        for name, results in results_dict.items():
            interactions_df = interactions_dict[name]
            report['detailed_analysis'][name] = self.analyze_bubble_formation(
                results, interactions_df
            )
        
        # Comparative analysis
        report['comparative_analysis'] = self.compare_bubble_prevention(
            results_dict, interactions_dict
        )
        
        # Statistical evidence
        report['statistical_evidence'] = self.metrics_calculator.calculate_comparative_metrics(
            results_dict, interactions_dict
        )
        
        # Recommendations
        report['recommendations'] = self._generate_recommendations(
            results_dict, interactions_dict
        )
        
        return report
    
    # Helper methods for bubble analysis
    
    def _analyze_diversity_timeline(self, results: SimulationResults) -> Dict[str, Any]:
        """Analyze the timeline of diversity changes."""
        timeline = {}
        
        if not results.content_diversity:
            return timeline
        
        diversity_values = results.content_diversity
        
        # Key metrics
        timeline['initial_diversity'] = diversity_values[0]
        timeline['final_diversity'] = diversity_values[-1]
        timeline['max_diversity'] = max(diversity_values)
        timeline['min_diversity'] = min(diversity_values)
        timeline['diversity_range'] = max(diversity_values) - min(diversity_values)
        
        # Critical points
        timeline['steepest_decline_point'] = self._find_steepest_decline(diversity_values)
        timeline['recovery_points'] = self._find_recovery_points(diversity_values)
        
        # Phases
        timeline['decline_phases'] = self._identify_decline_phases(diversity_values)
        
        return timeline
    
    def _analyze_user_bubble_susceptibility(self, interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze which users are most susceptible to filter bubbles."""
        if len(interactions_df) == 0:
            return {}
        
        segmentation = {}
        
        # Calculate diversity for each user
        user_diversity = {}
        for user_id in interactions_df['user_id'].unique():
            user_data = interactions_df[interactions_df['user_id'] == user_id]
            unique_items = user_data['item_id'].nunique()
            total_interactions = len(user_data)
            diversity = unique_items / max(1, total_interactions)
            user_diversity[user_id] = diversity
        
        # Segment users by diversity
        diversity_values = list(user_diversity.values())
        if diversity_values:
            low_threshold = np.percentile(diversity_values, 33)
            high_threshold = np.percentile(diversity_values, 67)
            
            low_diversity_users = [uid for uid, div in user_diversity.items() if div <= low_threshold]
            high_diversity_users = [uid for uid, div in user_diversity.items() if div >= high_threshold]
            
            segmentation['low_diversity_users'] = len(low_diversity_users)
            segmentation['high_diversity_users'] = len(high_diversity_users)
            segmentation['bubble_susceptible_fraction'] = len(low_diversity_users) / len(user_diversity)
        
        # Analyze user characteristics
        if 'user_exploration_factor' in interactions_df.columns:
            bubble_prone_data = interactions_df[interactions_df['user_id'].isin(low_diversity_users)]
            diverse_data = interactions_df[interactions_df['user_id'].isin(high_diversity_users)]
            
            segmentation['bubble_prone_characteristics'] = {
                'avg_exploration_factor': bubble_prone_data['user_exploration_factor'].mean(),
                'avg_position_bias': bubble_prone_data['user_position_bias_factor'].mean() if 'user_position_bias_factor' in bubble_prone_data.columns else 0.0
            }
            
            segmentation['diverse_user_characteristics'] = {
                'avg_exploration_factor': diverse_data['user_exploration_factor'].mean(),
                'avg_position_bias': diverse_data['user_position_bias_factor'].mean() if 'user_position_bias_factor' in diverse_data.columns else 0.0
            }
        
        return segmentation
    
    def _analyze_content_type_impact(self, interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how different content types contribute to bubble formation."""
        if len(interactions_df) == 0 or 'item_categories' not in interactions_df.columns:
            return {}
        
        content_impact = {}
        
        # Category concentration analysis
        category_counts = Counter()
        for cats in interactions_df['item_categories']:
            if isinstance(cats, list):
                category_counts.update(cats)
        
        total_category_interactions = sum(category_counts.values())
        
        if total_category_interactions > 0:
            content_impact['category_concentration'] = {
                cat: count / total_category_interactions 
                for cat, count in category_counts.most_common(10)
            }
        
        # Popularity bias analysis
        if 'item_popularity' in interactions_df.columns:
            popularity_quartiles = interactions_df['item_popularity'].quantile([0.25, 0.5, 0.75])
            
            q1_interactions = len(interactions_df[interactions_df['item_popularity'] <= popularity_quartiles[0.25]])
            q4_interactions = len(interactions_df[interactions_df['item_popularity'] >= popularity_quartiles[0.75]])
            
            content_impact['popularity_bias'] = {
                'low_popularity_share': q1_interactions / len(interactions_df),
                'high_popularity_share': q4_interactions / len(interactions_df),
                'popularity_concentration_ratio': q4_interactions / max(1, q1_interactions)
            }
        
        return content_impact
    
    def _identify_bubble_phases(self, results: SimulationResults) -> List[Dict[str, Any]]:
        """Identify distinct phases in bubble formation."""
        if not results.content_diversity:
            return []
        
        diversity_values = results.content_diversity
        phases = []
        
        # Simple phase detection based on trends
        window_size = max(3, len(diversity_values) // 5)
        
        for i in range(0, len(diversity_values) - window_size + 1, window_size):
            window = diversity_values[i:i + window_size]
            
            if len(window) > 1:
                slope, _, _, _, _ = stats.linregress(range(len(window)), window)
                
                phase = {
                    'start_timestep': i,
                    'end_timestep': i + len(window) - 1,
                    'start_diversity': window[0],
                    'end_diversity': window[-1],
                    'trend_slope': slope,
                    'phase_type': 'decline' if slope < -0.01 else 'stable' if abs(slope) <= 0.01 else 'recovery'
                }
                phases.append(phase)
        
        return phases
    
    def _calculate_bubble_score(self, results: SimulationResults) -> float:
        """Calculate an overall bubble formation score (higher = more bubble formation)."""
        if not results.content_diversity:
            return 0.0
        
        # Weighted combination of multiple factors
        diversity_loss = 1.0 - (results.content_diversity[-1] / max(0.001, results.content_diversity[0]))
        diversity_instability = np.var(results.content_diversity) if len(results.content_diversity) > 1 else 0.0
        
        # Combine factors
        bubble_score = 0.7 * diversity_loss + 0.3 * diversity_instability
        
        return max(0.0, bubble_score)
    
    def _calculate_diversity_preservation(self, results: SimulationResults) -> float:
        """Calculate how well diversity is preserved (higher = better preservation)."""
        if not results.content_diversity or len(results.content_diversity) < 2:
            return 0.0
        
        # Calculate average diversity and stability
        avg_diversity = np.mean(results.content_diversity)
        diversity_stability = 1.0 / (1.0 + np.var(results.content_diversity))
        
        # Combine factors
        preservation_score = 0.6 * avg_diversity + 0.4 * diversity_stability
        
        return preservation_score
    
    def _analyze_diversity_engagement_tradeoff(self, 
                                             results_dict: Dict[str, SimulationResults],
                                             interactions_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze the trade-off between diversity and engagement."""
        tradeoff_analysis = {}
        
        for name, results in results_dict.items():
            if results.content_diversity and results.engagement_rate:
                avg_diversity = np.mean(results.content_diversity)
                avg_engagement = np.mean(results.engagement_rate)
                
                tradeoff_analysis[name] = {
                    'avg_diversity': avg_diversity,
                    'avg_engagement': avg_engagement,
                    'efficiency_ratio': avg_diversity / max(0.001, avg_engagement),  # Diversity per unit engagement
                    'balanced_score': (avg_diversity + avg_engagement) / 2  # Simple balanced score
                }
        
        # Identify the best balanced approach
        if tradeoff_analysis:
            best_balanced = max(tradeoff_analysis.items(), 
                              key=lambda x: x[1]['balanced_score'])
            tradeoff_analysis['best_balanced_approach'] = best_balanced
        
        return tradeoff_analysis
    
    def _generate_executive_summary(self, 
                                  results_dict: Dict[str, SimulationResults],
                                  interactions_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate an executive summary of bubble analysis."""
        summary = {}
        
        # Key findings
        bubble_scores = {name: self._calculate_bubble_score(results) 
                        for name, results in results_dict.items()}
        
        summary['worst_bubble_formation'] = max(bubble_scores.items(), key=lambda x: x[1])
        summary['best_bubble_prevention'] = min(bubble_scores.items(), key=lambda x: x[1])
        
        # Diversity statistics
        diversity_stats = {}
        for name, results in results_dict.items():
            if results.content_diversity:
                diversity_stats[name] = {
                    'final_diversity': results.content_diversity[-1],
                    'diversity_loss': (results.content_diversity[0] - results.content_diversity[-1]) / max(0.001, results.content_diversity[0]) * 100
                }
        
        summary['diversity_statistics'] = diversity_stats
        
        # Business impact
        engagement_stats = {}
        for name, results in results_dict.items():
            if results.engagement_rate:
                engagement_stats[name] = {
                    'avg_engagement': np.mean(results.engagement_rate),
                    'engagement_stability': 1.0 / (1.0 + np.var(results.engagement_rate))
                }
        
        summary['engagement_statistics'] = engagement_stats
        
        return summary
    
    def _generate_recommendations(self, 
                                results_dict: Dict[str, SimulationResults],
                                interactions_dict: Dict[str, pd.DataFrame]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Analyze bubble scores
        bubble_scores = {name: self._calculate_bubble_score(results) 
                        for name, results in results_dict.items()}
        
        best_system = min(bubble_scores.items(), key=lambda x: x[1])[0]
        worst_system = max(bubble_scores.items(), key=lambda x: x[1])[0]
        
        recommendations.append(f"Use {best_system} approach for better bubble prevention")
        recommendations.append(f"Avoid {worst_system} approach due to high bubble formation")
        
        # Analyze diversity preservation
        for name, results in results_dict.items():
            if results.content_diversity and len(results.content_diversity) > 1:
                diversity_loss = (results.content_diversity[0] - results.content_diversity[-1]) / max(0.001, results.content_diversity[0])
                
                if diversity_loss > 0.5:  # More than 50% diversity loss
                    recommendations.append(f"Increase exploration in {name} to prevent severe diversity loss")
                elif diversity_loss < 0.1:  # Less than 10% diversity loss
                    recommendations.append(f"Consider {name} as a model for diversity preservation")
        
        # Business recommendations
        engagement_diversity_balance = {}
        for name, results in results_dict.items():
            if results.content_diversity and results.engagement_rate:
                avg_diversity = np.mean(results.content_diversity)
                avg_engagement = np.mean(results.engagement_rate)
                balance_score = min(avg_diversity, avg_engagement)  # Balanced score
                engagement_diversity_balance[name] = balance_score
        
        if engagement_diversity_balance:
            best_balanced = max(engagement_diversity_balance.items(), key=lambda x: x[1])[0]
            recommendations.append(f"For optimal business outcomes, consider {best_balanced} approach for best diversity-engagement balance")
        
        return recommendations
    
    # Additional helper methods
    
    def _find_steepest_decline(self, values: List[float]) -> Optional[int]:
        """Find the timestep with the steepest diversity decline."""
        if len(values) < 2:
            return None
        
        max_decline = 0
        steepest_point = None
        
        for i in range(1, len(values)):
            decline = values[i-1] - values[i]
            if decline > max_decline:
                max_decline = decline
                steepest_point = i
        
        return steepest_point
    
    def _find_recovery_points(self, values: List[float]) -> List[int]:
        """Find timesteps where diversity recovers."""
        recovery_points = []
        
        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                recovery_points.append(i)
        
        return recovery_points
    
    def _identify_decline_phases(self, values: List[float]) -> List[Dict[str, int]]:
        """Identify continuous decline phases."""
        phases = []
        start = None
        
        for i in range(1, len(values)):
            if values[i] < values[i-1]:  # Declining
                if start is None:
                    start = i-1
            else:  # Not declining
                if start is not None:
                    phases.append({'start': start, 'end': i-1})
                    start = None
        
        # Handle case where decline continues to the end
        if start is not None:
            phases.append({'start': start, 'end': len(values)-1})
        
        return phases