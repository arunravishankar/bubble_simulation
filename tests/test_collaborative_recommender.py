import numpy as np
from bubble_simulation.recommenders.collaborative import CollaborativeFilteringRecommender
from bubble_simulation.content.item import Item
from bubble_simulation.users.user import User

# Helper functions to create test data
def create_test_item(item_id, categories=None, features=None):
    if categories is None:
        categories = [item_id % 3]
    if features is None:
        features = np.zeros(10)
        features[item_id % 5] = 1.0  # More overlap in features than in other tests
    return Item(
        item_id=item_id, 
        features=features, 
        categories=categories
    )

def create_test_user(user_id, preference_vector=None):
    if preference_vector is None:
        preference_vector = np.zeros(10)
        preference_vector[user_id % 5] = 1.0  # Match feature dimension pattern
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
def test_cf_recommender_initialization():
    """Test that the collaborative filtering recommender initializes correctly."""
    recommender = CollaborativeFilteringRecommender(
        name="CFTest", 
        retrain_frequency=5, 
        top_k=10,
        num_factors=20,
        regularization=0.1,
        use_implicit=True
    )
    assert recommender.name == "CFTest"
    assert recommender.retrain_frequency == 5
    assert recommender.top_k == 10
    assert recommender.num_factors == 20
    assert recommender.regularization == 0.1
    assert recommender.use_implicit is True
    
    # Model parameters should be None before training
    assert recommender.user_factors is None
    assert recommender.item_factors is None
    assert recommender.user_bias is None
    assert recommender.item_bias is None
    assert recommender.global_bias == 0.0


def test_cf_training_basic():
    """Test basic collaborative filtering training with minimal data."""
    recommender = CollaborativeFilteringRecommender(num_factors=2)
    
    # Create minimal test data
    items = [create_test_item(i) for i in range(3)]
    users = [create_test_user(i) for i in range(2)]
    
    # Create a simple interaction pattern:
    # User 0 likes items 0 and 1
    # User 1 likes items 1 and 2
    interactions = [
        create_test_interaction(user_id=0, item_id=0, engaged=True),
        create_test_interaction(user_id=0, item_id=1, engaged=True),
        create_test_interaction(user_id=0, item_id=2, engaged=False),
        create_test_interaction(user_id=1, item_id=0, engaged=False),
        create_test_interaction(user_id=1, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=2, engaged=True),
    ]
    
    # Train the model
    recommender.train(interactions, items, users)
    
    # Check that model parameters were created
    assert recommender.user_factors is not None
    assert recommender.item_factors is not None
    assert recommender.user_factors.shape == (2, 2)  # 2 users, 2 factors
    assert recommender.item_factors.shape == (3, 2)  # 3 items, 2 factors
    
    # Mappings should include all users and items
    assert len(recommender.user_id_map) == 2
    assert len(recommender.item_id_map) == 3
    
    # Training metrics should be recorded
    metrics = recommender.get_metrics()
    assert 'train_time' in metrics
    assert len(metrics['train_time']) == 1


def test_cf_recommendations():
    """Test that recommendations are generated correctly."""
    recommender = CollaborativeFilteringRecommender(num_factors=2, top_k=2)
    
    # Create test data
    items = [create_test_item(i) for i in range(3)]
    users = [create_test_user(i) for i in range(2)]
    
    # Create a simple interaction pattern:
    # User 0 likes items 0 and 1
    # User 1 likes items 1 and 2
    interactions = [
        create_test_interaction(user_id=0, item_id=0, engaged=True),
        create_test_interaction(user_id=0, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=2, engaged=True),
    ]
    
    # Train the model
    recommender.train(interactions, items, users)
    
    # For user 0, item 2 should be recommended (similar to item 1)
    # For user 1, item 0 should be recommended (similar to item 1)
    
    # Get recommendations for user 0
    user0 = users[0]
    user0.interaction_history = [items[0], items[1]]  # Set interaction history
    recommendations_0 = recommender.recommend(user0, items)
    assert len(recommendations_0) == 1  # Only item 2 is available (0 and 1 are in history)
    assert recommendations_0[0].item_id == 2
    
    # Get recommendations for user 1
    user1 = users[1]
    user1.interaction_history = [items[1], items[2]]  # Set interaction history
    recommendations_1 = recommender.recommend(user1, items)
    assert len(recommendations_1) == 1  # Only item 0 is available (1 and 2 are in history)
    assert recommendations_1[0].item_id == 0


def test_cf_with_exclusions():
    """Test that recommendations respect exclusions."""
    recommender = CollaborativeFilteringRecommender(num_factors=2)
    
    # Create test data
    items = [create_test_item(i) for i in range(4)]
    users = [create_test_user(i) for i in range(2)]
    
    # Create a simple interaction pattern
    interactions = [
        create_test_interaction(user_id=0, item_id=0, engaged=True),
        create_test_interaction(user_id=0, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=2, engaged=True),
    ]
    
    # Train the model
    recommender.train(interactions, items, users)
    
    # User 0 has interacted with items 0 and 1
    user0 = users[0]
    user0.interaction_history = [items[0], items[1]]
    
    # Exclude item 2 as well
    exclude_items = {2}
    
    # Get recommendations for user 0
    recommendations = recommender.recommend(user0, items, exclude_items=exclude_items)
    
    # Only item 3 should be recommended (0, 1 in history, 2 explicitly excluded)
    assert len(recommendations) == 1
    assert recommendations[0].item_id == 3


def test_cf_similar_items():
    """Test finding similar items."""
    recommender = CollaborativeFilteringRecommender(num_factors=2)
    
    # Create items with specific patterns
    items = [
        create_test_item(0, features=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        create_test_item(1, features=np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),  # Similar to item 0
        create_test_item(2, features=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),  # Different
    ]
    
    # Create users and interactions
    users = [create_test_user(i) for i in range(2)]
    interactions = [
        create_test_interaction(user_id=0, item_id=0, engaged=True),
        create_test_interaction(user_id=0, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=2, engaged=True),
    ]
    
    # Train the model
    recommender.train(interactions, items, users)
    
    # Get similar items to item 0
    similar_items = recommender.get_most_similar_items(items[0], n=2, all_items=items)
    
    # Item 1 should be most similar to item 0
    assert len(similar_items) > 0
    similar_item, similarity = similar_items[0]
    assert similar_item.item_id == 1
    
    # Get similar items to item 2
    similar_items = recommender.get_most_similar_items(items[2], n=2, all_items=items)
    
    # Both item 0 and 1 should be less similar to item 2
    if len(similar_items) > 0:
        similar_item, similarity = similar_items[0]
        # The similarity should be lower than between items 0 and 1
        assert similarity < 0.9


def test_cf_similar_users():
    """Test finding similar users."""
    recommender = CollaborativeFilteringRecommender(num_factors=2)
    
    # Create users with specific patterns
    users = [
        create_test_user(0, preference_vector=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        create_test_user(1, preference_vector=np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),  # Similar to user 0
        create_test_user(2, preference_vector=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),  # Different
    ]
    
    # Create items and some interactions
    items = [create_test_item(i) for i in range(5)]
    interactions = [
        create_test_interaction(user_id=0, item_id=0, engaged=True),
        create_test_interaction(user_id=0, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=0, engaged=True),
        create_test_interaction(user_id=1, item_id=1, engaged=True),
        create_test_interaction(user_id=2, item_id=2, engaged=True),
        create_test_interaction(user_id=2, item_id=3, engaged=True),
    ]
    
    # Train the model
    recommender.train(interactions, items, users)
    
    # Get similar users to user 0
    similar_users = recommender.get_most_similar_users(users[0], users, n=2)
    
    # User 1 should be most similar to user 0
    assert len(similar_users) > 0
    similar_user, similarity = similar_users[0]
    assert similar_user.user_id == 1
    
    # User 2 should be less similar to user 0
    # Get similar users again including user 2
    if len(similar_users) > 1:
        similar_user, similarity = similar_users[1]
        assert similar_user.user_id == 2


def test_cf_untrained_recommendations():
    """Test recommendation behavior when model hasn't been trained."""
    recommender = CollaborativeFilteringRecommender()
    
    # Create some test data
    items = [create_test_item(i) for i in range(3)]
    user = create_test_user(0)
    
    # Without training, recommendations should still work
    recommendations = recommender.recommend(user, items)
    
    # Should return a list (probably random or default order)
    assert isinstance(recommendations, list)
    assert len(recommendations) <= len(items)


def test_cf_new_user_recommendations():
    """Test recommendations for users not seen during training."""
    recommender = CollaborativeFilteringRecommender(num_factors=2)
    
    # Create initial data for training
    items = [create_test_item(i) for i in range(5)]
    training_users = [create_test_user(i) for i in range(2)]
    interactions = [
        create_test_interaction(user_id=0, item_id=0, engaged=True),
        create_test_interaction(user_id=0, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=2, engaged=True),
        create_test_interaction(user_id=1, item_id=3, engaged=True),
    ]
    
    # Train the model
    recommender.train(interactions, items, training_users)
    
    # Create a new user not seen during training
    new_user = create_test_user(99)
    
    # The new user has interacted with item 0
    new_user.interaction_history = [items[0]]
    
    # Get recommendations
    recommendations = recommender.recommend(new_user, items)
    
    # Should return some recommendations
    assert len(recommendations) > 0
    
    # Item 0 should not be in recommendations (already in history)
    assert all(item.item_id != 0 for item in recommendations)


def test_update_retraining_schedule():
    """Test that the recommender retrains according to schedule."""
    recommender = CollaborativeFilteringRecommender(retrain_frequency=5)
    
    # Create test data
    items = [create_test_item(i) for i in range(10)]
    users = [create_test_user(i) for i in range(5)]
    interactions = [
        create_test_interaction(user_id=0, item_id=1),
        create_test_interaction(user_id=1, item_id=2),
        create_test_interaction(user_id=2, item_id=3)
    ]
    
    # Initially, model hasn't been trained
    assert recommender.user_factors is None
    
    # Initial update should trigger training
    was_retrained = recommender.update(interactions, items, users, timestep=0)
    assert was_retrained is True
    assert recommender.user_factors is not None
    assert recommender.last_training_step == 0
    
    # Store the trained factors
    original_factors = recommender.user_factors.copy()
    
    # Update within retrain frequency should not trigger training
    was_retrained = recommender.update(interactions, items, users, timestep=4)
    assert was_retrained is False
    assert np.array_equal(recommender.user_factors, original_factors)
    assert recommender.last_training_step == 0
    
    # Add more interactions
    more_interactions = interactions + [
        create_test_interaction(user_id=3, item_id=4),
        create_test_interaction(user_id=4, item_id=5)
    ]
    
    # Update at retrain frequency should trigger training
    was_retrained = recommender.update(more_interactions, items, users, timestep=5)
    assert was_retrained is True
    assert not np.array_equal(recommender.user_factors, original_factors)
    assert recommender.last_training_step == 5