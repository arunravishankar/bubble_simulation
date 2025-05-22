import numpy as np
from typing import List, Dict, Set
from bubble_simulation.recommenders.base import BaseRecommender
from bubble_simulation.content.item import Item
from bubble_simulation.users.user import User


# Create a concrete implementation of BaseRecommender for testing
class TestableRecommender(BaseRecommender):
    """Simple implementation of BaseRecommender for testing."""
    
    def __init__(self, name="TestRecommender", retrain_frequency=10, top_k=5):
        super().__init__(name=name, retrain_frequency=retrain_frequency, top_k=top_k)
        self.was_trained = False
    
    def train(self, interactions: List[Dict], items: List[Item], users: List[User]) -> None:
        """Mark that training occurred."""
        self.was_trained = True
        self.add_metric('train_loss', 0.5)  # Dummy metric
        
    def recommend(self, 
                 user: User, 
                 items: List[Item], 
                 n: int = None,
                 exclude_items: Set[int] = None) -> List[Item]:
        """Return top n items based on item_id (for deterministic testing)."""
        filtered_items = self.filter_items(items, exclude_items)
        sorted_items = sorted(filtered_items, key=lambda x: x.item_id)
        return sorted_items[:n if n is not None else self.top_k]


# Helper functions to create test data
def create_test_item(item_id, categories=None, features=None):
    if categories is None:
        categories = [item_id % 3]
    if features is None:
        features = np.zeros(10)
        features[item_id % 10] = 1.0
    return Item(item_id=item_id, features=features, categories=categories)

def create_test_user(user_id):
    preference_vector = np.zeros(10)
    preference_vector[user_id % 10] = 1.0
    return User(user_id=user_id, preference_vector=preference_vector)

def create_test_interaction(user_id, item_id, engaged=True, position=0, timestamp=0):
    return {
        'user_id': user_id,
        'item_id': item_id,
        'engaged': engaged,
        'position': position,
        'timestamp': timestamp
    }


# Test cases
def test_base_recommender_initialization():
    """Test that a recommender can be properly initialized."""
    recommender = TestableRecommender(name="TestReco", retrain_frequency=5, top_k=10)
    assert recommender.name == "TestReco"
    assert recommender.retrain_frequency == 5
    assert recommender.top_k == 10
    assert recommender.last_training_step == -1
    assert recommender.training_count == 0
    assert 'train_time' in recommender.metrics
    assert 'inference_time' in recommender.metrics
    assert 'train_loss' in recommender.metrics


def test_update_retraining_schedule():
    """Test that the recommender retrains according to schedule."""
    recommender = TestableRecommender(retrain_frequency=5)
    
    # Create some test data
    items = [create_test_item(i) for i in range(10)]
    users = [create_test_user(i) for i in range(5)]
    interactions = [
        create_test_interaction(user_id=0, item_id=1),
        create_test_interaction(user_id=1, item_id=2),
        create_test_interaction(user_id=2, item_id=3)
    ]
    
    # Initial update should trigger training
    was_retrained = recommender.update(interactions, items, users, timestep=0)
    assert was_retrained is True
    assert recommender.was_trained is True
    assert recommender.last_training_step == 0
    
    # Reset training flag for next test
    recommender.was_trained = False
    
    # Update within retrain frequency should not trigger training
    was_retrained = recommender.update(interactions, items, users, timestep=4)
    assert was_retrained is False
    assert recommender.was_trained is False
    assert recommender.last_training_step == 0
    
    # Update at retrain frequency should trigger training
    was_retrained = recommender.update(interactions, items, users, timestep=5)
    assert was_retrained is True
    assert recommender.was_trained is True
    assert recommender.last_training_step == 5


def test_filter_items():
    """Test that filter_items correctly filters excluded items."""
    recommender = TestableRecommender()
    
    # Create test items
    items = [create_test_item(i) for i in range(10)]
    
    # Test with no exclusions
    filtered = recommender.filter_items(items)
    assert len(filtered) == 10
    assert set(item.item_id for item in filtered) == set(range(10))
    
    # Test with exclusions
    exclude_ids = {1, 3, 5, 7}
    filtered = recommender.filter_items(items, exclude_ids)
    assert len(filtered) == 6
    assert set(item.item_id for item in filtered) == {0, 2, 4, 6, 8, 9}


def test_recommend_with_exclusions():
    """Test that recommendation respects excluded items."""
    recommender = TestableRecommender(top_k=5)
    
    # Create test data
    items = [create_test_item(i) for i in range(10)]
    user = create_test_user(0)
    
    # Test recommendations with no exclusions
    recommendations = recommender.recommend(user, items)
    assert len(recommendations) == 5
    assert [item.item_id for item in recommendations] == [0, 1, 2, 3, 4]
    
    # Test with exclusions
    exclude_ids = {1, 3}
    recommendations = recommender.recommend(user, items, exclude_items=exclude_ids)
    assert len(recommendations) == 5
    assert [item.item_id for item in recommendations] == [0, 2, 4, 5, 6]
    
    # Test with n parameter
    recommendations = recommender.recommend(user, items, n=3)
    assert len(recommendations) == 3
    assert [item.item_id for item in recommendations] == [0, 1, 2]


def test_metrics_tracking():
    """Test that metrics are properly tracked."""
    recommender = TestableRecommender()
    
    # Create test data
    items = [create_test_item(i) for i in range(10)]
    users = [create_test_user(i) for i in range(5)]
    interactions = [
        create_test_interaction(user_id=0, item_id=1),
        create_test_interaction(user_id=1, item_id=2)
    ]
    
    # Trigger training
    recommender.train(interactions, items, users)
    
    # Check that training metrics were added
    metrics = recommender.get_metrics()
    assert 'train_loss' in metrics
    assert len(metrics['train_loss']) == 1
    assert metrics['train_loss'][0] == 0.5
    
    # Add a custom metric
    recommender.add_metric('custom_metric', 0.75)
    
    # Check that custom metric was added
    metrics = recommender.get_metrics()
    assert 'custom_metric' in metrics
    assert len(metrics['custom_metric']) == 1
    assert metrics['custom_metric'][0] == 0.75
    
    # Add another value to existing metric
    recommender.add_metric('custom_metric', 0.85)
    metrics = recommender.get_metrics()
    assert len(metrics['custom_metric']) == 2
    assert metrics['custom_metric'][1] == 0.85


def test_user_item_interactions_conversion():
    """Test conversion of interactions to user-item mapping."""
    recommender = TestableRecommender()
    
    # Create test interactions
    interactions = [
        create_test_interaction(user_id=0, item_id=1, engaged=True),
        create_test_interaction(user_id=0, item_id=2, engaged=False),
        create_test_interaction(user_id=1, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=3, engaged=True)
    ]
    
    # Convert to user-item mapping
    user_item_map = recommender._get_user_item_interactions(interactions)
    
    # Check structure
    assert 0 in user_item_map
    assert 1 in user_item_map
    
    # Check user 0's interactions
    assert user_item_map[0][1] == 1.0  # engaged
    assert user_item_map[0][2] == 0.0  # not engaged
    
    # Check user 1's interactions
    assert user_item_map[1][1] == 1.0  # engaged
    assert user_item_map[1][3] == 1.0  # engaged