import numpy as np
from bubble_simulation.recommenders.popularity import PopularityRecommender
from bubble_simulation.content.item import Item
from bubble_simulation.users.user import User

# Helper functions to create test data
def create_test_item(item_id, categories=None, features=None, popularity_score=None):
    if categories is None:
        categories = [item_id % 3]
    if features is None:
        features = np.zeros(10)
        features[item_id % 10] = 1.0
    if popularity_score is None:
        # Items with even IDs are more popular by default,
        # and add a small variation between different even IDs (and between different odd IDs)
        base_score = 0.8 if item_id % 2 == 0 else 0.2
        # Add a small variation based on the item ID to make each one unique
        popularity_score = base_score - (item_id * 0.01)
    return Item(
        item_id=item_id, 
        features=features, 
        categories=categories, 
        popularity_score=popularity_score
    )

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
def test_popularity_recommender_initialization():
    """Test that a popularity recommender can be properly initialized."""
    recommender = PopularityRecommender(
        name="PopularityTest", 
        retrain_frequency=5, 
        top_k=10,
        recency_weight=0.6
    )
    assert recommender.name == "PopularityTest"
    assert recommender.retrain_frequency == 5
    assert recommender.top_k == 10
    assert recommender.recency_weight == 0.6
    assert recommender.item_popularity == {}
    assert recommender.interaction_counts == {}


def test_popularity_training():
    """Test that the popularity recommender correctly calculates item popularity."""
    recommender = PopularityRecommender(recency_weight=0.5)
    
    # Create test data
    items = [create_test_item(i) for i in range(5)]
    users = [create_test_user(i) for i in range(3)]
    
    # No interactions yet - should use intrinsic popularity
    recommender.train([], items, users)
    
    # Check that popularity scores were set
    assert len(recommender.item_popularity) == 5
    
    # Items with even IDs have higher intrinsic popularity
    assert recommender.item_popularity[0] > recommender.item_popularity[1]
    assert recommender.item_popularity[2] > recommender.item_popularity[3]
    
    # Now add some interactions
    interactions = [
        # Item 1 (odd ID, intrinsically less popular) gets lots of engagement
        create_test_interaction(user_id=0, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=1, engaged=True),
        create_test_interaction(user_id=2, item_id=1, engaged=True),
        
        # Item 0 (even ID, intrinsically more popular) gets some engagement
        create_test_interaction(user_id=0, item_id=0, engaged=True),
        
        # Item 2 gets no engagement
        create_test_interaction(user_id=0, item_id=2, engaged=False),
    ]
    
    # Retrain with interactions
    recommender.train(interactions, items, users)
    
    # Item 1 should now be more popular than Item 0 due to engagement
    assert recommender.item_popularity[1] > recommender.item_popularity[0]
    
    # Item 2 should still be more popular than items 3,4 due to intrinsic popularity
    assert recommender.item_popularity[2] > recommender.item_popularity[3]
    assert recommender.item_popularity[2] > recommender.item_popularity[4]


def test_recommend_by_popularity():
    """Test that the recommender recommends the most popular items."""
    recommender = PopularityRecommender(top_k=3)
    
    # Create items with explicit popularity scores
    items = [
        create_test_item(0, popularity_score=0.1),  # Least popular
        create_test_item(1, popularity_score=0.5),  # Medium popularity
        create_test_item(2, popularity_score=0.9),  # Most popular
        create_test_item(3, popularity_score=0.7),  # Second most popular
        create_test_item(4, popularity_score=0.3),  # Fourth most popular
    ]
    
    # Train without interactions, using only intrinsic popularity
    recommender.train([], items, [])
    
    # Create a test user
    user = create_test_user(0)
    
    # Get recommendations
    recommendations = recommender.recommend(user, items)
    
    # Should recommend the top 3 most popular items
    assert len(recommendations) == 3
    assert recommendations[0].item_id == 2  # Most popular
    assert recommendations[1].item_id == 3  # Second most popular
    assert recommendations[2].item_id == 1  # Third most popular


def test_recommend_with_exclusions():
    """Test that recommendations exclude items the user has already interacted with."""
    recommender = PopularityRecommender(top_k=3)
    
    # Create items with explicit popularity scores
    items = [
        create_test_item(0, popularity_score=0.1),  # Least popular
        create_test_item(1, popularity_score=0.5),  # Medium popularity
        create_test_item(2, popularity_score=0.9),  # Most popular
        create_test_item(3, popularity_score=0.7),  # Second most popular
        create_test_item(4, popularity_score=0.3),  # Fourth most popular
    ]
    
    # Train without interactions
    recommender.train([], items, [])
    
    # Create a user who has already interacted with items 2 and 3
    user = create_test_user(0)
    user.interaction_history = [items[2], items[3]]
    
    # Get recommendations
    recommendations = recommender.recommend(user, items)
    
    # Should exclude items 2 and 3, recommending the next most popular
    assert len(recommendations) == 3
    assert recommendations[0].item_id == 1  # Now most popular available
    assert recommendations[1].item_id == 4  # Now second most popular available
    assert recommendations[2].item_id == 0  # Now third most popular available


def test_additional_exclusions():
    """Test that additional exclusions are respected."""
    recommender = PopularityRecommender(top_k=3)
    
    # Create items with explicit popularity scores
    items = [
        create_test_item(0, popularity_score=0.1),  # Least popular
        create_test_item(1, popularity_score=0.5),  # Medium popularity
        create_test_item(2, popularity_score=0.9),  # Most popular
        create_test_item(3, popularity_score=0.7),  # Second most popular
        create_test_item(4, popularity_score=0.3),  # Fourth most popular
    ]
    
    # Train without interactions
    recommender.train([], items, [])
    
    # Create a user with no interactions
    user = create_test_user(0)
    
    # Exclude some items specifically
    exclude_items = {1, 3}
    
    # Get recommendations
    recommendations = recommender.recommend(user, items, exclude_items=exclude_items)
    
    # Should exclude items 1 and 3
    assert len(recommendations) == 3
    assert recommendations[0].item_id == 2  # Most popular
    assert recommendations[1].item_id == 4  # Now second most popular available
    assert recommendations[2].item_id == 0  # Now third most popular available


def test_recency_weight_impact():
    """Test that the recency weight parameter affects recommendations."""
    # Create items with explicit popularity scores
    items = [
        create_test_item(0, popularity_score=0.9),  # Most intrinsically popular
        create_test_item(1, popularity_score=0.1),  # Least intrinsically popular
    ]
    
    # Create interactions where the least intrinsically popular item gets engagement
    interactions = [
        create_test_interaction(user_id=0, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=1, engaged=True),
    ]
    
    # Test with high recency weight (favors engagement)
    recommender_high_recency = PopularityRecommender(recency_weight=0.9)
    recommender_high_recency.train(interactions, items, [])
    
    # Test with low recency weight (favors intrinsic popularity)
    recommender_low_recency = PopularityRecommender(recency_weight=0.1)
    recommender_low_recency.train(interactions, items, [])
    
    # With high recency weight, item 1 should be more popular
    assert recommender_high_recency.item_popularity[1] > recommender_high_recency.item_popularity[0]
    
    # With low recency weight, item 0 should still be more popular
    assert recommender_low_recency.item_popularity[0] > recommender_low_recency.item_popularity[1]


def test_update_retraining():
    """Test that the recommender retrains according to schedule."""
    recommender = PopularityRecommender(retrain_frequency=5)
    
    # Create test data
    items = [create_test_item(i) for i in range(5)]
    users = [create_test_user(i) for i in range(3)]
    
    # Initial state - no popularity scores
    assert recommender.item_popularity == {}
    
    # Update at timestep 0 should trigger training
    was_retrained = recommender.update([], items, users, timestep=0)
    assert was_retrained is True
    assert len(recommender.item_popularity) == 5
    assert recommender.last_training_step == 0
    
    # Store the initial popularity scores
    initial_scores = recommender.item_popularity.copy()
    
    # Add some interactions
    interactions = [
        create_test_interaction(user_id=0, item_id=1, engaged=True),
        create_test_interaction(user_id=1, item_id=1, engaged=True),
    ]
    
    # Update within retrain frequency should not trigger training
    was_retrained = recommender.update(interactions, items, users, timestep=4)
    assert was_retrained is False
    assert recommender.item_popularity == initial_scores  # Unchanged
    
    # Update at retrain frequency should trigger training
    was_retrained = recommender.update(interactions, items, users, timestep=5)
    assert was_retrained is True
    assert recommender.item_popularity != initial_scores  # Changed
    assert recommender.last_training_step == 5


def test_empty_recommendations():
    """Test behavior when all items are excluded."""
    recommender = PopularityRecommender()
    
    # Create test data
    items = [create_test_item(i) for i in range(3)]
    user = create_test_user(0)
    
    # Train the recommender
    recommender.train([], items, [])
    
    # Exclude all items
    exclude_items = {0, 1, 2}
    
    # Should return empty list when all items are excluded
    recommendations = recommender.recommend(user, items, exclude_items=exclude_items)
    assert len(recommendations) == 0


def test_performance_metrics():
    """Test that performance metrics are tracked."""
    recommender = PopularityRecommender()
    
    # Create test data
    items = [create_test_item(i) for i in range(5)]
    users = [create_test_user(i) for i in range(3)]
    
    # Measure training time
    recommender.train([], items, users)
    
    # Check that training time was recorded
    metrics = recommender.get_metrics()
    assert 'train_time' in metrics
    assert len(metrics['train_time']) == 1
    assert metrics['train_time'][0] > 0  # Should be positive
    
    # Measure inference time
    user = create_test_user(0)
    recommender.recommend(user, items)
    
    # Check that inference time was recorded
    metrics = recommender.get_metrics()
    assert 'inference_time' in metrics
    assert len(metrics['inference_time']) == 1
    assert metrics['inference_time'][0] > 0  # Should be positive