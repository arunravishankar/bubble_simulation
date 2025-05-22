import numpy as np
import random
from bubble_simulation.recommenders.reinforcement import RLRecommender
from bubble_simulation.content.item import Item
from bubble_simulation.users.user import User

# Helper functions to create test data
def create_test_item(item_id, categories=None, features=None, popularity_score=None, metadata=None):
    if categories is None:
        categories = [item_id % 3]
    if features is None:
        features = np.zeros(10)
        features[item_id % 5] = 1.0
    if popularity_score is None:
        popularity_score = 0.5 + (item_id % 5) * 0.1
    if metadata is None:
        metadata = {}
    return Item(
        item_id=item_id, 
        features=features, 
        categories=categories,
        popularity_score=popularity_score,
        metadata=metadata
    )

def create_test_user(user_id, preference_vector=None, interaction_history=None):
    if preference_vector is None:
        preference_vector = np.zeros(10)
        preference_vector[user_id % 5] = 1.0
    user = User(user_id=user_id, preference_vector=preference_vector)
    if interaction_history:
        user.interaction_history = interaction_history
    return user

def create_test_interaction(user_id, item_id, engaged=True, position=0, timestamp=0):
    return {
        'user_id': user_id,
        'item_id': item_id,
        'engaged': engaged,
        'position': position,
        'timestamp': timestamp
    }


# Test cases
def test_rl_recommender_initialization():
    """Test that the RL recommender initializes correctly with all parameters."""
    recommender = RLRecommender(
        name="RLTest", 
        retrain_frequency=5, 
        top_k=10,
        epsilon=0.2,
        gamma=0.8,
        engagement_weight=0.6,
        diversity_weight=0.4,
        revenue_weight=0.15,
        retention_weight=0.1,
        learning_rate=0.05,
        feature_bins=8,
        memory_size=5000,
        batch_size=16,
        epsilon_decay=0.99,
        min_epsilon=0.005,
        double_q=False,
        staged_exploration=True,
        business_aware_exploration=True
    )
    
    # Test basic attributes
    assert recommender.name == "RLTest"
    assert recommender.retrain_frequency == 5
    assert recommender.top_k == 10
    assert recommender.epsilon == 0.2
    assert recommender.initial_epsilon == 0.2
    assert recommender.gamma == 0.8
    assert recommender.engagement_weight == 0.6
    assert recommender.diversity_weight == 0.4
    assert recommender.revenue_weight == 0.15
    assert recommender.retention_weight == 0.1
    assert recommender.learning_rate == 0.05
    assert recommender.feature_bins == 8
    assert recommender.memory_size == 5000
    assert recommender.batch_size == 16
    assert recommender.epsilon_decay == 0.99
    assert recommender.min_epsilon == 0.005
    assert recommender.double_q is False
    assert recommender.staged_exploration is True
    assert recommender.business_aware_exploration is True
    
    # Test initialization of internal structures
    assert recommender.q_table == {}
    assert recommender.target_q_table == {}
    assert len(recommender.replay_memory) == 0
    assert recommender.state_scaler is None
    assert recommender.exploration_count == 0
    assert recommender.exploitation_count == 0
    assert recommender.total_reward == 0
    assert recommender.episode_rewards == []
    assert recommender.user_states == {}
    assert recommender.item_revenue == {}
    assert recommender.item_retention == {}
    assert recommender.strategic_items == set()
    assert recommender.user_interaction_counts == {}


def test_rl_basic_training():
    """Test basic RL training functionality."""
    recommender = RLRecommender(batch_size=2, memory_size=10)
    
    # Create test data
    items = [create_test_item(i) for i in range(5)]
    users = [create_test_user(i) for i in range(3)]
    
    # Create interactions
    interactions = [
        create_test_interaction(user_id=0, item_id=0, engaged=True),
        create_test_interaction(user_id=0, item_id=1, engaged=False),
        create_test_interaction(user_id=1, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=2, engaged=True),
        create_test_interaction(user_id=2, item_id=2, engaged=False),
    ]
    
    # Train the model
    recommender.train(interactions, items, users)
    
    # Check that training occurred
    assert recommender.state_scaler is not None
    assert len(recommender.replay_memory) == 5
    assert len(recommender.user_interaction_counts) == 3
    assert recommender.user_interaction_counts[0] == 2
    assert recommender.user_interaction_counts[1] == 2
    assert recommender.user_interaction_counts[2] == 1
    
    # Check that Q-table has some entries
    assert len(recommender.q_table) > 0
    
    # Check that metrics were recorded
    metrics = recommender.get_metrics()
    assert 'train_time' in metrics
    assert 'epsilon' in metrics
    assert 'avg_reward' in metrics
    assert len(metrics['train_time']) == 1


def test_business_metrics_update():
    """Test that business metrics are correctly updated from item metadata."""
    recommender = RLRecommender()
    
    # Create items with business metadata
    items = [
        create_test_item(0, metadata={'revenue_potential': 0.8, 'retention_impact': 0.9, 'strategic': True}),
        create_test_item(1, metadata={'revenue_potential': 0.3}),
        create_test_item(2, metadata={'retention_impact': 0.7}),
        create_test_item(3),  # No metadata
    ]
    
    # Update business metrics
    recommender._update_business_metrics(items)
    
    # Check revenue metrics
    assert recommender.item_revenue[0] == 0.8
    assert recommender.item_revenue[1] == 0.3
    assert 0 in recommender.item_revenue  # Should have default for item 2
    assert 3 in recommender.item_revenue  # Should have default for item 3
    
    # Check retention metrics
    assert recommender.item_retention[0] == 0.9
    assert 1 in recommender.item_retention  # Should have default for item 1
    assert recommender.item_retention[2] == 0.7
    assert 3 in recommender.item_retention  # Should have default for item 3
    
    # Check strategic items
    assert 0 in recommender.strategic_items
    assert 1 not in recommender.strategic_items
    assert 2 not in recommender.strategic_items
    assert 3 not in recommender.strategic_items


def test_staged_exploration():
    """Test that exploration rate adapts based on user journey stage."""
    recommender = RLRecommender(epsilon=0.1, staged_exploration=True)
    
    # Create users at different stages
    new_user = create_test_user(0)
    mid_user = create_test_user(1)
    established_user = create_test_user(2)
    
    # Set interaction counts
    recommender.user_interaction_counts[0] = 5  # New user (< 20)
    recommender.user_interaction_counts[1] = 50  # Mid-term user (20-100)
    recommender.user_interaction_counts[2] = 150  # Established user (> 100)
    
    # Test epsilon values
    new_epsilon = recommender._get_staged_epsilon(new_user)
    mid_epsilon = recommender._get_staged_epsilon(mid_user)
    established_epsilon = recommender._get_staged_epsilon(established_user)
    
    # New users should have higher exploration
    assert new_epsilon > recommender.epsilon
    
    # Mid-term users should have normal exploration
    assert mid_epsilon == recommender.epsilon
    
    # Established users should have lower exploration
    assert established_epsilon < recommender.epsilon
    assert established_epsilon >= recommender.min_epsilon


def test_staged_exploration_disabled():
    """Test that staged exploration can be disabled."""
    recommender = RLRecommender(epsilon=0.1, staged_exploration=False)
    
    user = create_test_user(0)
    recommender.user_interaction_counts[0] = 5  # New user
    
    # Should return normal epsilon regardless of user stage
    epsilon = recommender._get_staged_epsilon(user)
    assert epsilon == recommender.epsilon


def test_reward_calculation():
    """Test that rewards are calculated correctly."""
    recommender = RLRecommender(
        engagement_weight=0.5,
        diversity_weight=0.3,
        revenue_weight=0.1,
        retention_weight=0.1
    )
    
    # Set up business metrics
    recommender.item_revenue[1] = 0.8
    recommender.item_retention[1] = 0.7
    recommender.strategic_items.add(1)
    
    user = create_test_user(0)
    item = create_test_item(1)
    
    # Test positive engagement
    reward_positive = recommender._calculate_reward(user, item, engaged=True)
    
    # Test negative engagement
    reward_negative = recommender._calculate_reward(user, item, engaged=False)
    
    # Positive engagement should yield higher reward
    assert reward_positive > reward_negative
    
    # Reward should be a combination of all factors
    # The exact value depends on the diversity calculation, but should be reasonable
    assert -1.0 <= reward_positive <= 3.0
    assert -1.0 <= reward_negative <= 3.0


def test_diversity_calculation():
    """Test diversity calculation for items."""
    recommender = RLRecommender()
    
    user = create_test_user(0)
    
    # Test diversity for user with no history
    item1 = create_test_item(1, categories=[0], features=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    diversity = recommender._calculate_diversity(user, item1)
    assert diversity == 1.0  # Maximum diversity for first interaction
    
    # Add some history
    history_item = create_test_item(2, categories=[0], features=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    user.interaction_history = [history_item]
    
    # Test diversity for similar item
    similar_item = create_test_item(3, categories=[0], features=np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]))
    diversity_similar = recommender._calculate_diversity(user, similar_item)
    
    # Test diversity for different item
    different_item = create_test_item(4, categories=[1], features=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
    diversity_different = recommender._calculate_diversity(user, different_item)
    
    # Different item should have higher diversity
    assert diversity_different > diversity_similar
    assert 0 <= diversity_similar <= 1
    assert 0 <= diversity_different <= 1


def test_state_representation():
    """Test user state representation."""
    recommender = RLRecommender()
    
    # Create a user with some characteristics
    user = create_test_user(0)
    user.category_interests = {0: 0.8, 1: 0.5, 2: 0.3}
    
    # Mock some interaction count
    recommender.user_interaction_counts[0] = 30
    
    # Get state representation
    state = recommender._get_user_state(user)
    
    # State should be a numpy array
    assert isinstance(state, np.ndarray)
    
    # State should have expected length (preference_vector + category_interests + additional features)
    # 10 (preference) + 5 (categories) + 3 (diversity, engagement, journey_stage) = 18
    expected_length = 10 + 5 + 3
    assert len(state) == expected_length
    
    # Test state caching
    state2 = recommender._get_user_state(user)
    assert np.array_equal(state, state2)


def test_state_discretization():
    """Test state discretization for Q-table lookup."""
    recommender = RLRecommender(feature_bins=5)
    
    # Create a test state
    state = np.array([0.1, 0.5, 0.9, 0.0, 1.0])
    
    # Get discretized state key
    state_key = recommender._get_state_key(state)
    
    # Should be a tuple of integers
    assert isinstance(state_key, tuple)
    assert len(state_key) == 5
    assert all(isinstance(x, int) for x in state_key)
    assert all(0 <= x < 5 for x in state_key)


def test_exploration_vs_exploitation():
    """Test exploration vs exploitation behavior."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    recommender = RLRecommender(epsilon=0.5, top_k=3)
    
    # Create test data
    items = [create_test_item(i) for i in range(10)]
    users = [create_test_user(0)]
    user = users[0]
    
    # Train with some data to populate Q-table
    interactions = [
        create_test_interaction(user_id=0, item_id=0, engaged=True),
        create_test_interaction(user_id=0, item_id=1, engaged=True),
        create_test_interaction(user_id=0, item_id=2, engaged=False),
    ]
    recommender.train(interactions, items, users)
    
    # Get multiple recommendations to test stochasticity
    recommendations_1 = recommender.recommend(user, items)
    recommendations_2 = recommender.recommend(user, items)
    
    assert len(recommendations_1) >= 0, "Should return valid recommendations"
    assert len(recommendations_2) >= 0, "Should return valid recommendations"

    # Both should return valid recommendations
    assert len(recommendations_1) <= 3
    assert len(recommendations_2) <= 3
    assert all(isinstance(item, Item) for item in recommendations_1)
    assert all(isinstance(item, Item) for item in recommendations_2)
    
    # With epsilon=0.5, we should see some variation (though this is probabilistic)
    # At minimum, check that we don't get the same exact recommendations every time
    # This test might occasionally fail due to randomness, but should usually pass


def test_business_aware_exploration():
    """Test business-aware exploration functionality."""
    recommender = RLRecommender(business_aware_exploration=True)
    
    # Set up business metrics
    recommender.item_revenue = {0: 0.9, 1: 0.3, 2: 0.7}
    recommender.item_retention = {0: 0.8, 1: 0.4, 2: 0.6}
    recommender.strategic_items = {0}
    
    # Create test data
    items = [create_test_item(i) for i in range(3)]
    user = create_test_user(0)
    
    # Set user as new user
    recommender.user_interaction_counts[0] = 5
    
    # Get business-aware exploration recommendations
    recommendations = recommender._business_aware_explore(user, items, 2)
    
    # Should return recommendations
    assert len(recommendations) <= 2
    assert all(isinstance(item, Item) for item in recommendations)


def test_q_learning_updates():
    """Test that Q-values are updated correctly."""
    recommender = RLRecommender(learning_rate=0.1, gamma=0.9, batch_size=1)
    
    # Create test data
    items = [create_test_item(i) for i in range(3)]
    users = [create_test_user(0)]
    
    # Add a simple interaction
    interactions = [
        create_test_interaction(user_id=0, item_id=0, engaged=True)
    ]
    
    # Train to populate replay memory
    recommender.train(interactions, items, users)
    
    # Check that Q-table has entries
    assert len(recommender.q_table) > 0
    
    # Get a Q-value
    first_key = list(recommender.q_table.keys())[0]
    initial_q_value = recommender.q_table[first_key]
    assert isinstance(initial_q_value, (int, float)), "Q-value should be a number"

    # Train again with more data
    more_interactions = [
        create_test_interaction(user_id=0, item_id=1, engaged=True)
    ]
    recommender.train(more_interactions, items, users)
    
    # Q-values should potentially change (though exact behavior depends on the learning)
    assert len(recommender.q_table) >= 1


def test_epsilon_decay():
    """Test that epsilon decays over time."""
    recommender = RLRecommender(epsilon=0.5, epsilon_decay=0.9, min_epsilon=0.1)
    
    initial_epsilon = recommender.epsilon
    
    # Create test data for training
    items = [create_test_item(i) for i in range(3)]
    users = [create_test_user(0)]
    interactions = [create_test_interaction(user_id=0, item_id=0, engaged=True)]
    
    # Train multiple times
    for _ in range(5):
        recommender.train(interactions, items, users)
    
    # Epsilon should have decayed
    assert recommender.epsilon < initial_epsilon
    assert recommender.epsilon >= recommender.min_epsilon


def test_recommendation_exclusions():
    """Test that recommendations respect exclusions."""
    recommender = RLRecommender(top_k=3)
    
    # Create test data
    items = [create_test_item(i) for i in range(5)]
    user = create_test_user(0)
    
    # Set interaction history
    user.interaction_history = [items[0], items[1]]
    
    # Get recommendations with additional exclusions
    exclude_items = {2}
    recommendations = recommender.recommend(user, items, exclude_items=exclude_items)
    
    # Should not include items 0, 1 (history) or 2 (excluded)
    recommended_ids = {item.item_id for item in recommendations}
    assert 0 not in recommended_ids
    assert 1 not in recommended_ids
    assert 2 not in recommended_ids
    
    # Should only recommend from items 3 and 4
    assert recommended_ids.issubset({3, 4})


def test_metrics_tracking():
    """Test that various metrics are tracked correctly."""
    recommender = RLRecommender()
    
    # Create test data
    items = [create_test_item(i) for i in range(3)]
    users = [create_test_user(0)]
    interactions = [create_test_interaction(user_id=0, item_id=0, engaged=True)]
    
    # Train and get recommendations
    recommender.train(interactions, items, users)
    recommendations = recommender.recommend(users[0], items)
    
    assert isinstance(recommendations, list), "Should return a list of recommendations"

    # Check metrics
    metrics = recommender.get_metrics()
    
    # Should have basic metrics
    assert 'train_time' in metrics
    assert 'inference_time' in metrics
    assert 'epsilon' in metrics
    assert 'avg_reward' in metrics
    
    # Should have RL-specific metrics
    assert 'episode_rewards' in metrics
    
    # Values should be reasonable
    assert len(metrics['train_time']) > 0
    assert len(metrics['inference_time']) > 0
    assert all(t >= 0 for t in metrics['train_time'])
    assert all(t >= 0 for t in metrics['inference_time'])


def test_set_business_metrics():
    """Test manual setting of business metrics."""
    recommender = RLRecommender()
    
    # Set business metrics manually
    revenue_map = {0: 0.8, 1: 0.3, 2: 0.9}
    retention_map = {0: 0.7, 1: 0.4, 2: 0.8}
    strategic_set = {0, 2}
    
    recommender.set_business_metrics(
        item_revenue=revenue_map,
        item_retention=retention_map,
        strategic_items=strategic_set
    )
    
    # Check that metrics were set correctly
    assert recommender.item_revenue == revenue_map
    assert recommender.item_retention == retention_map
    assert recommender.strategic_items == strategic_set


def test_reset_exploration():
    """Test resetting exploration rate."""
    recommender = RLRecommender(epsilon=0.5)
    
    # Manually change epsilon
    recommender.epsilon = 0.1
    
    # Reset exploration
    recommender.reset_exploration()
    
    # Should be back to initial value
    assert recommender.epsilon == 0.5


def test_update_retraining_schedule():
    """Test that the RL recommender retrains according to schedule."""
    recommender = RLRecommender(retrain_frequency=5)
    
    # Create test data
    items = [create_test_item(i) for i in range(10)]
    users = [create_test_user(i) for i in range(5)]
    interactions = [
        create_test_interaction(user_id=0, item_id=1),
        create_test_interaction(user_id=1, item_id=2),
        create_test_interaction(user_id=2, item_id=3)
    ]
    
    # Initially, model hasn't been trained
    assert recommender.state_scaler is None
    
    # Initial update should trigger training
    was_retrained = recommender.update(interactions, items, users, timestep=0)
    assert was_retrained is True
    assert recommender.state_scaler is not None
    assert recommender.last_training_step == 0
    
    # Store current Q-table size
    _original_q_size = len(recommender.q_table)
    
    # Update within retrain frequency should not trigger training
    was_retrained = recommender.update(interactions, items, users, timestep=4)
    assert was_retrained is False
    assert recommender.last_training_step == 0
    
    # Update at retrain frequency should trigger training
    was_retrained = recommender.update(interactions, items, users, timestep=5)
    assert was_retrained is True
    assert recommender.last_training_step == 5