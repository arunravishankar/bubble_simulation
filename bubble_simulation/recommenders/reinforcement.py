import numpy as np
import time
import logging
import random
from collections import deque
from typing import List, Dict, Optional, Set
from sklearn.preprocessing import StandardScaler
from ..content.item import Item
from ..users.user import User
from .base import BaseRecommender
from ..settings import (
    RECO_TOP_K, RL_EXPLORATION_EPSILON, RL_DISCOUNT_FACTOR,
    RL_REWARD_ENGAGEMENT_WEIGHT, RL_REWARD_DIVERSITY_WEIGHT, 
    RL_REWARD_REVENUE_WEIGHT, RL_REWARD_RETENTION_WEIGHT,
    RL_LEARNING_RATE, RL_FEATURE_BINS, RL_MEMORY_SIZE,
    RL_BATCH_SIZE, RL_EPSILON_DECAY, RL_MIN_EPSILON,
    RL_STRATEGIC_BONUS, RL_ENABLE_STAGED_EXPLORATION,
    RL_ENABLE_BUSINESS_AWARE_EXPLORATION,
    RL_NEW_USER_THRESHOLD, RL_ESTABLISHED_USER_THRESHOLD,
    RL_NEW_USER_EPSILON_BOOST, RL_ESTABLISHED_USER_EPSILON_REDUCTION,
    RL_HISTORY_WINDOW_SIZE, RL_CATEGORY_INTEREST_SIZE,
    RL_STATE_SAMPLE_SIZE, RL_HIGH_RETENTION_CATEGORIES,
    RL_BASE_RETENTION_SCORE, RL_RETENTION_CATEGORY_BONUS,
    RL_REVENUE_POPULARITY_FACTOR
)
logger = logging.getLogger(__name__)

class RLRecommender(BaseRecommender):
    """
    Reinforcement Learning based recommender system.
    
    This recommender uses Q-learning to learn a policy for recommending items
    that balances immediate engagement with long-term user satisfaction and
    content diversity.
    
    Key features:
    - Balances exploration and exploitation via epsilon-greedy policy
    - Adapts exploration rate based on user journey stage
    - Considers diversity, engagement, and business metrics in rewards
    - Uses experience replay for more stable learning
    - Supports prioritized business-aware exploration
    """
    
    def __init__(self, 
                 name: str = "RLRecommender",
                 retrain_frequency: int = 10,
                 top_k: int = RECO_TOP_K,
                 epsilon: float = RL_EXPLORATION_EPSILON,
                 gamma: float = RL_DISCOUNT_FACTOR,
                 engagement_weight: float = RL_REWARD_ENGAGEMENT_WEIGHT,
                 diversity_weight: float = RL_REWARD_DIVERSITY_WEIGHT,
                 revenue_weight: float = RL_REWARD_REVENUE_WEIGHT,  
                 retention_weight: float = RL_REWARD_RETENTION_WEIGHT, 
                 learning_rate: float = RL_LEARNING_RATE,
                 feature_bins: int = RL_FEATURE_BINS,
                 memory_size: int = RL_MEMORY_SIZE,
                 batch_size: int = RL_BATCH_SIZE,
                 epsilon_decay: float = RL_EPSILON_DECAY,
                 min_epsilon: float = RL_MIN_EPSILON,
                 double_q: bool = True,
                 staged_exploration: bool = RL_ENABLE_STAGED_EXPLORATION,  
                 business_aware_exploration: bool = RL_ENABLE_BUSINESS_AWARE_EXPLORATION):
        """
        Initialize the RL recommender.
        
        Args:
            name: Identifier for this recommender instance
            retrain_frequency: How often to retrain the model (in timesteps)
            top_k: Default number of items to recommend
            epsilon: Exploration rate for epsilon-greedy policy
            gamma: Discount factor for future rewards
            engagement_weight: Weight of engagement in reward function
            diversity_weight: Weight of diversity in reward function
            revenue_weight: Weight of revenue in reward function
            retention_weight: Weight of retention in reward function
            learning_rate: Learning rate for Q-value updates
            feature_bins: Number of bins for state discretization
            memory_size: Size of replay memory
            batch_size: Batch size for training
            epsilon_decay: Rate at which epsilon decays over time
            min_epsilon: Minimum exploration rate
            double_q: Whether to use double Q-learning
            staged_exploration: Whether to adjust exploration rate based on user journey
            business_aware_exploration: Whether to prioritize business metrics in exploration
        """
        super().__init__(name=name, retrain_frequency=retrain_frequency, top_k=top_k)
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.engagement_weight = engagement_weight
        self.diversity_weight = diversity_weight
        self.revenue_weight = revenue_weight
        self.retention_weight = retention_weight
        self.learning_rate = learning_rate
        self.feature_bins = feature_bins
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.double_q = double_q
        self.staged_exploration = staged_exploration
        self.business_aware_exploration = business_aware_exploration
        
        # Initialize Q-tables and model parameters
        self.q_table = {}  # State-action to Q-value mapping
        self.target_q_table = {}  # For double Q-learning
        self.replay_memory = deque(maxlen=memory_size)
        self.state_scaler = None  # For normalizing state features
        
        # Track statistics for debugging and evaluation
        self.exploration_count = 0
        self.exploitation_count = 0
        self.total_reward = 0
        self.episode_rewards = []
        
        # Keep a mapping of user states
        self.user_states = {}  # User ID to current state
        
        # Business metrics
        self.item_revenue = {}  # Item ID to revenue mapping
        self.item_retention = {}  # Item ID to retention impact mapping
        self.strategic_items = set()  # Set of items with strategic importance
        
        # User journey stages
        self.user_interaction_counts = {}  # User ID to interaction count
        
    def train(self, interactions: List[Dict], items: List[Item], users: List[User]) -> None:
        """
        Train the RL model using batch updates from replay memory.
        
        Args:
            interactions: List of user-item interaction dictionaries
            items: List of available content items
            users: List of users
        """
        start_time = time.time()
        
        # Initialize scaler for state features if not already done
        if self.state_scaler is None:
            self._initialize_state_scaler(users, items)
        
        # Update business metrics for items if available
        self._update_business_metrics(items)
        
        # Add latest interactions to replay memory
        for interaction in interactions:
            user_id = interaction['user_id']
            item_id = interaction['item_id']
            engaged = interaction['engaged']
            
            # Find the user and item objects
            user = next((u for u in users if u.user_id == user_id), None)
            item = next((i for i in items if i.item_id == item_id), None)
            
            if user is None or item is None:
                continue
            
            # Update user interaction count for staged exploration
            if user_id not in self.user_interaction_counts:
                self.user_interaction_counts[user_id] = 0
            self.user_interaction_counts[user_id] += 1
                
            # Calculate reward based on engagement, diversity, and business metrics
            reward = self._calculate_reward(user, item, engaged)
            
            # Get the current state and action
            state = self._get_user_state(user)
            action = item_id  # The action is recommending a specific item
            
            # Get the next state (after interaction)
            next_state = self._get_user_state(user)  # Updated state after interaction
            
            # Store transition in replay memory
            self.replay_memory.append((state, action, reward, next_state, user_id))
            
            # Update exploration/exploitation counts
            # Assume exploration if interaction was from a recommendation made with
            # epsilon probability
            if random.random() < self.epsilon:
                self.exploration_count += 1
            else:
                self.exploitation_count += 1
                
            # Update total reward
            self.total_reward += reward
        
        # Only train if we have enough samples
        if len(self.replay_memory) >= self.batch_size:
            losses = []
            
            # Perform multiple training iterations
            for _ in range(min(10, len(interactions))):
                loss = self._train_batch()
                losses.append(loss)
            
            # Record average loss
            if losses:
                self.add_metric('train_loss', np.mean(losses))
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Calculate average reward for this training session
        avg_reward = self.total_reward / max(1, self.exploration_count + self.exploitation_count)
        self.episode_rewards.append(avg_reward)
        
        # Reset counters for next episode
        self.exploration_count = 0
        self.exploitation_count = 0
        self.total_reward = 0
        
        # Log training metrics
        train_time = time.time() - start_time
        self.add_metric('train_time', train_time)
        self.add_metric('epsilon', self.epsilon)
        self.add_metric('avg_reward', avg_reward)
        
        # Copy main Q-table to target for double Q-learning
        if self.double_q:
            self.target_q_table = self.q_table.copy()
            
        # Increment training count
        self.training_count += 1
        
        logger.info(f"Trained {self.name} in {train_time:.4f} seconds, "
                   f"avg reward: {avg_reward:.4f}, epsilon: {self.epsilon:.4f}")
    
    def _update_business_metrics(self, items: List[Item]) -> None:
        """
        Update business metrics for items based on their metadata.
        
        Args:
            items: List of available content items
        """
        for item in items:
            # Check if item has revenue information in metadata
            if 'revenue_potential' in item.metadata:
                self.item_revenue[item.item_id] = item.metadata['revenue_potential']
            elif item.item_id not in self.item_revenue:
                # Default revenue based on popularity as a fallback
                self.item_revenue[item.item_id] = item.popularity_score * RL_REVENUE_POPULARITY_FACTOR                
            # Check if item has retention impact in metadata
            if 'retention_impact' in item.metadata:
                self.item_retention[item.item_id] = item.metadata['retention_impact']
            elif item.item_id not in self.item_retention:
                # Default retention impact based on categories as a fallback
                # Assume certain categories have higher retention impact
                category_impact = sum(1 for cat in item.categories if cat in RL_HIGH_RETENTION_CATEGORIES)
                self.item_retention[item.item_id] = min(1.0, RL_BASE_RETENTION_SCORE + RL_RETENTION_CATEGORY_BONUS * category_impact)
                
            # Check if item is marked as strategic
            if 'strategic' in item.metadata and item.metadata['strategic']:
                self.strategic_items.add(item.item_id)
    
    def _train_batch(self) -> float:
        """
        Train on a batch of experiences from replay memory.
        
        Returns:
            Loss value for this batch
        """
        # Sample a batch of transitions from replay memory
        if len(self.replay_memory) < self.batch_size:
            return 0.0
            
        batch = random.sample(self.replay_memory, self.batch_size)
        
        total_loss = 0.0
        
        # Update Q-values for each transition in the batch
        for state, action, reward, next_state, user_id in batch:
            # Get current Q-value
            state_key = self._get_state_key(state)
            if (state_key, action) not in self.q_table:
                self.q_table[(state_key, action)] = 0.0
            current_q = self.q_table[(state_key, action)]
            
            # Get next action according to current policy
            next_action = self._get_best_action(next_state)
            
            # Get next Q-value from target network (for double Q-learning)
            next_state_key = self._get_state_key(next_state)
            if self.double_q:
                if (next_state_key, next_action) not in self.target_q_table:
                    self.target_q_table[(next_state_key, next_action)] = 0.0
                next_q = self.target_q_table[(next_state_key, next_action)]
            else:
                if (next_state_key, next_action) not in self.q_table:
                    self.q_table[(next_state_key, next_action)] = 0.0
                next_q = self.q_table[(next_state_key, next_action)]
            
            # Calculate target Q-value
            target_q = reward + self.gamma * next_q
            
            # Calculate loss
            loss = (target_q - current_q) ** 2
            total_loss += loss
            
            # Update Q-value
            self.q_table[(state_key, action)] += self.learning_rate * (target_q - current_q)
        
        # Return average loss
        return total_loss / self.batch_size
    
    def recommend(self, 
                 user: User, 
                 items: List[Item], 
                 n: int = None,
                 exclude_items: Optional[Set[int]] = None) -> List[Item]:
        """
        Generate recommendations for a user based on learned Q-values.
        
        Uses epsilon-greedy policy: with probability epsilon, explores by 
        recommending randomly; otherwise exploits by recommending items
        with highest Q-values.
        
        Args:
            user: User to recommend items for
            items: Available items to recommend from
            n: Number of items to recommend
            exclude_items: Set of item IDs to exclude from recommendations
            
        Returns:
            List of recommended items
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
            
        # Get user's current state
        state = self._get_user_state(user)
        
        # Adjust epsilon based on user journey stage if enabled
        effective_epsilon = self._get_staged_epsilon(user)
        
        # Decide whether to explore or exploit
        if random.random() < effective_epsilon:
            # Explore: recommend based on exploration strategy
            if self.business_aware_exploration:
                recommendations = self._business_aware_explore(user, available_items, n)
            else:
                recommendations = self._explore(user, available_items, n)
            self.exploration_count += 1
        else:
            # Exploit: recommend based on Q-values
            recommendations = self._exploit(state, available_items, n)
            self.exploitation_count += 1
            
        # Update metrics
        inference_time = time.time() - start_time
        self.add_metric('inference_time', inference_time)
        
        return recommendations
    
    def _get_staged_epsilon(self, user: User) -> float:
        """
        Get exploration rate based on user's journey stage.
        
        New users get higher exploration, while established users get lower exploration.
        
        Args:
            user: User to get exploration rate for
            
        Returns:
            Adjusted epsilon value
        """
        if not self.staged_exploration:
            return self.epsilon  # Use global epsilon if staged exploration is disabled
            
        # Get user's interaction count
        interaction_count = self.user_interaction_counts.get(user.user_id, 0)
        
        # Stage 1: High initial exploration for new users
        if interaction_count < RL_NEW_USER_THRESHOLD:
            return min(1.0, self.epsilon * RL_NEW_USER_EPSILON_BOOST)
            
        # Stage 2: Targeted mid-term exploration
        elif interaction_count < RL_ESTABLISHED_USER_THRESHOLD:
            return self.epsilon  # Use normal epsilon
            
        # Stage 3: Maintenance exploration for established users
        else:
            return max(self.min_epsilon, self.epsilon * RL_ESTABLISHED_USER_EPSILON_REDUCTION)
    
    def _explore(self, user: User, available_items: List[Item], n: int) -> List[Item]:
        """
        Explore by recommending somewhat diverse items.
        
        Args:
            user: User to recommend for
            available_items: Items available for recommendation
            n: Number of items to recommend
            
        Returns:
            List of recommended items
        """
        # Start with a random sample of items that's larger than needed
        sample_size = min(len(available_items), n * 3)
        candidate_items = random.sample(available_items, sample_size)
        
        # Calculate diversity score for each candidate
        diversity_scores = []
        for item in candidate_items:
            diversity = self._calculate_diversity(user, item)
            diversity_scores.append((item, diversity))
            
        # Sort by diversity (higher is better)
        diversity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top n diverse items
        return [item for item, _ in diversity_scores[:n]]
    
    def _business_aware_explore(self, user: User, available_items: List[Item], n: int) -> List[Item]:
        """
        Explore with awareness of business objectives.
        
        Balances diversity with revenue potential, retention impact, and strategic priorities.
        
        Args:
            user: User to recommend for
            available_items: Items available for recommendation
            n: Number of items to recommend
            
        Returns:
            List of recommended items
        """
        # Start with a random sample of items that's larger than needed
        sample_size = min(len(available_items), n * 3)
        candidate_items = random.sample(available_items, sample_size)
        
        # Calculate combined score for each candidate
        # Score combines diversity with business metrics
        combined_scores = []
        for item in candidate_items:
            # Get diversity score
            diversity = self._calculate_diversity(user, item)
            
            # Get business metrics
            revenue = self.item_revenue.get(item.item_id, 0.0)
            retention = self.item_retention.get(item.item_id, 0.0)
            
            # Apply strategic bonus
            strategic_bonus = RL_STRATEGIC_BONUS if item.item_id in self.strategic_items else 0.0
            
            # Interaction count determines balance between diversity and business metrics
            interaction_count = self.user_interaction_counts.get(user.user_id, 0)
            
            if interaction_count < RL_NEW_USER_THRESHOLD:
                # New users: prioritize diversity and retention
                score = 0.5 * diversity + 0.1 * revenue + 0.3 * retention + strategic_bonus
            elif interaction_count < RL_ESTABLISHED_USER_THRESHOLD:
                # Mid-term users: balanced approach
                score = 0.3 * diversity + 0.3 * revenue + 0.3 * retention + strategic_bonus
            else:
                # Established users: prioritize revenue and strategic items
                score = 0.2 * diversity + 0.4 * revenue + 0.2 * retention + strategic_bonus * 1.5
                
            combined_scores.append((item, score))
            
        # Sort by combined score (higher is better)
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top n items
        return [item for item, _ in combined_scores[:n]]
    
    def _exploit(self, state: np.ndarray, available_items: List[Item], n: int) -> List[Item]:
        """
        Exploit by recommending items with highest Q-values.
        
        Args:
            state: User's current state
            available_items: Items available for recommendation
            n: Number of items to recommend
            
        Returns:
            List of recommended items
        """
        state_key = self._get_state_key(state)
        
        # Calculate Q-value for each available item
        q_values = []
        for item in available_items:
            action = item.item_id
            q_value = self.q_table.get((state_key, action), 0.0)
            q_values.append((item, q_value))
            
        # Sort by Q-value (higher is better)
        q_values.sort(key=lambda x: x[1], reverse=True)
        
        # Take top n items
        return [item for item, _ in q_values[:n]]
    
    def _get_user_state(self, user: User) -> np.ndarray:
        """
        Get the current state representation for a user.
        
        The state includes user preferences, interaction history features,
        and diversity metrics to provide context for decision-making.
        
        Args:
            user: User to get state for
            
        Returns:
            State vector representation
        """
        # If we've already calculated this user's state, return it
        if user.user_id in self.user_states:
            return self.user_states[user.user_id]
            
        # Extract user preference vector
        pref_vector = user.preference_vector
        
        # Extract category interests
        category_interests = list(user.category_interests.values())
        # Pad with zeros if needed
        while len(category_interests) < RL_CATEGORY_INTEREST_SIZE:  # Using a fixed size for categories
            category_interests.append(0.0)
        # Or truncate if too many
        category_interests = category_interests[:RL_CATEGORY_INTEREST_SIZE]        
        # Calculate diversity of recent history
        history_diversity = user.get_diversity_metric()
        
        # Calculate average engagement rate
        engagement_rate = user.get_engagement_rate()
        
        # User journey stage (from interaction count)
        interaction_count = self.user_interaction_counts.get(user.user_id, 0)
        journey_stage = min(1.0, interaction_count / RL_ESTABLISHED_USER_THRESHOLD)  # Normalize to [0, 1]
        
        # Combine all features into state vector
        state_features = np.concatenate([
            pref_vector,
            category_interests,
            [history_diversity, engagement_rate, journey_stage]
        ])
        
        # Scale features
        if self.state_scaler is not None:
            state_features = self.state_scaler.transform([state_features])[0]
            
        # Store for reuse
        self.user_states[user.user_id] = state_features
        
        return state_features
    
    def _get_state_key(self, state: np.ndarray) -> tuple:
        """
        Convert continuous state vector to discrete key for Q-table lookup.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Tuple key for Q-table
        """
        # Discretize each dimension of the state using binning
        discretized = []
        for feature in state:
            bin_idx = min(self.feature_bins - 1, 
                         max(0, int(feature * self.feature_bins)))
            discretized.append(bin_idx)
            
        return tuple(discretized)
    
    def _get_best_action(self, state: np.ndarray) -> int:
        """
        Get the best action (item) for a given state according to Q-values.
        
        Args:
            state: User's current state
            
        Returns:
            Item ID of best action
        """
        state_key = self._get_state_key(state)
        
        # Find all actions (items) we have Q-values for this state
        q_values = {}
        for (s, a), q in self.q_table.items():
            if s == state_key:
                q_values[a] = q
                
        # If no actions found, return a default value
        if not q_values:
            return -1
            
        # Return action with highest Q-value
        return max(q_values.items(), key=lambda x: x[1])[0]
    
    def _calculate_reward(self, user: User, item: Item, engaged: bool) -> float:
        """
        Calculate reward for a user-item interaction.
        
        Combines immediate engagement with diversity contribution and business metrics
        for a more balanced reward signal.
        
        Args:
            user: User who interacted
            item: Item that was interacted with
            engaged: Whether the user engaged positively
            
        Returns:
            Calculated reward value
        """
        # Base reward from engagement
        engagement_reward = 1.0 if engaged else -0.2
        
        # Calculate diversity contribution
        diversity_contribution = self._calculate_diversity(user, item)
        
        # Get business metrics
        revenue = self.item_revenue.get(item.item_id, 0.0)
        retention = self.item_retention.get(item.item_id, 0.0)
        
        # Apply strategic bonus
        strategic_bonus = RL_STRATEGIC_BONUS if item.item_id in self.strategic_items else 0.0
        
        # Combine rewards using weights
        reward = (
            self.engagement_weight * engagement_reward + 
            self.diversity_weight * diversity_contribution +
            self.revenue_weight * revenue +
            self.retention_weight * retention +
            strategic_bonus
        )
        
        return reward
    
    def _calculate_diversity(self, user: User, item: Item) -> float:
        """
        Calculate how much an item contributes to diversity.
        
        Higher values mean the item is different from the user's recent history.
        
        Args:
            user: User to calculate for
            item: Item to evaluate
            
        Returns:
            Diversity contribution score
        """
        if not user.interaction_history:
            return 1.0  # Maximum diversity for first interaction
            
        # Limit to recent history
        recent_history = user.interaction_history[-RL_HISTORY_WINDOW_SIZE:] if len(user.interaction_history) > RL_HISTORY_WINDOW_SIZE else user.interaction_history
        
        # Calculate category diversity
        history_categories = set()
        for hist_item in recent_history:
            history_categories.update(hist_item.categories)
            
        item_categories = set(item.categories)
        new_categories = item_categories - history_categories
        
        # Proportion of new categories
        if item_categories:
            category_diversity = len(new_categories) / len(item_categories)
        else:
            category_diversity = 0.0
            
        # Calculate feature diversity (average distance to history items)
        similarities = []
        for hist_item in recent_history:
            sim = item.similarity(hist_item)
            similarities.append(sim)
            
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            feature_diversity = 1.0 - avg_similarity
        else:
            feature_diversity = 1.0
            
        # Combine category and feature diversity
        return 0.4 * category_diversity + 0.6 * feature_diversity
    
    def _initialize_state_scaler(self, users: List[User], items: List[Item]) -> None:
        """
        Initialize scaler for normalizing state features.
        
        Args:
            users: List of users to sample states from
            items: List of items (unused, but kept for consistency)
        """
        # Sample states from a subset of users
        sample_size = min(len(users), RL_STATE_SAMPLE_SIZE)
        sample_users = random.sample(users, sample_size)
        
        # Get state features for each sampled user
        states = []
        for user in sample_users:
            # Temporarily clear user_states to force recalculation
            if user.user_id in self.user_states:
                del self.user_states[user.user_id]
                
            state = self._get_user_state(user)
            states.append(state)
            
        # Fit scaler on sample states
        if states:
            self.state_scaler = StandardScaler().fit(states)
    
    def reset_exploration(self) -> None:
        """Reset epsilon to initial value to encourage exploration."""
        self.epsilon = self.initial_epsilon
        
    def set_business_metrics(self, 
                           item_revenue: Dict[int, float] = None, 
                           item_retention: Dict[int, float] = None,
                           strategic_items: Set[int] = None) -> None:
        """
        Set business metrics for items directly.
        
        Args:
            item_revenue: Mapping of item ID to revenue potential
            item_retention: Mapping of item ID to retention impact
            strategic_items: Set of strategically important item IDs
        """
        if item_revenue is not None:
            self.item_revenue = item_revenue
            
        if item_retention is not None:
            self.item_retention = item_retention
            
        if strategic_items is not None:
            self.strategic_items = strategic_items
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Get performance metrics for this recommender.
        
        Returns:
            Dictionary of metrics
        """
        metrics = super().get_metrics()
        
        # Add RL-specific metrics
        if hasattr(self, 'episode_rewards') and self.episode_rewards:
            metrics['episode_rewards'] = self.episode_rewards
            
        return metrics
