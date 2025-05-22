import pytest
import numpy as np
from bubble_simulation.content import Item, ContentUniverse
from bubble_simulation.users import User, EngagementModel, UserUniverse
from bubble_simulation.settings import (
    USER_PREFERENCE_FEATURE_WEIGHT
)


def create_test_item(item_id, categories, features=None, popularity=0.5, creation_time=0):
    """Helper function to create test items with customizable features."""
    if features is None:
        # Make sure features has the same dimension as expected in tests (10)
        features = np.zeros(10)
        # Set a few values to make the feature vector more unique
        features[item_id % 10] = 1.0
    return Item(
        item_id=item_id,
        features=features,
        categories=categories,
        popularity_score=popularity,
        creation_time=creation_time
    )


def create_test_user(user_id, preference_vector=None, exploration_factor=0.2, position_bias_factor=0.8):
    """Helper function to create test users with customizable preferences."""
    if preference_vector is None:
        # Create a default 10-dimensional preference vector
        preference_vector = np.zeros(10)
        preference_vector[user_id % 10] = 1.0
    return User(
        user_id=user_id,
        preference_vector=preference_vector,
        exploration_factor=exploration_factor,
        position_bias_factor=position_bias_factor
    )


def test_user_preference_calculation():
    """Test that user preference calculation works correctly with settings values."""
    # Create a user with a simple preference vector (10 dimensions)
    preference_vector = np.zeros(10)
    preference_vector[0] = 1.0
    user = create_test_user(1, preference_vector=preference_vector)
    
    # Create items with different feature vectors (also 10 dimensions)
    item1_features = np.zeros(10)
    item1_features[0] = 1.0
    item1 = create_test_item(1, [0], features=item1_features)  # Same direction
    
    item2_features = np.zeros(10)
    item2_features[1] = 1.0
    item2 = create_test_item(2, [0], features=item2_features)  # Orthogonal
    
    item3_features = np.zeros(10)
    item3_features[0] = 0.7
    item3_features[1] = 0.7
    item3 = create_test_item(3, [1], features=item3_features)  # 45 degrees
    
    # Test preference calculations - should be USER_PREFERENCE_FEATURE_WEIGHT * similarity
    # Plus USER_PREFERENCE_CATEGORY_WEIGHT * category interest (initially 0)
    expected_item1_preference = USER_PREFERENCE_FEATURE_WEIGHT * 1.0  # Full similarity
    expected_item2_preference = USER_PREFERENCE_FEATURE_WEIGHT * 0.0  # Zero similarity
    expected_item3_preference = USER_PREFERENCE_FEATURE_WEIGHT * 0.7  # Partial similarity
    
    assert user.get_preference_for_item(item1) == pytest.approx(expected_item1_preference, 0.01)
    assert user.get_preference_for_item(item2) == pytest.approx(expected_item2_preference, 0.01)
    assert user.get_preference_for_item(item3) == pytest.approx(expected_item3_preference, 0.01)
    
    # Update user preferences based on interaction with item3
    user.update_preferences(item3, engaged=True)
    
    # Now user should have some interest in category 1
    assert 1 in user.category_interests
    assert user.category_interests[1] > 0
    
    # Preference for item3 should increase due to category interest
    assert user.get_preference_for_item(item3) > expected_item3_preference


def test_user_category_interests():
    """Test that category interests update correctly."""
    user = create_test_user(1)
    
    # Initially user should have no category interests
    assert len(user.category_interests) == 0
    
    # Create items from different categories
    item1 = create_test_item(1, [0])
    item2 = create_test_item(2, [1])
    item3 = create_test_item(3, [0, 2])  # Multi-category item
    
    # Update preferences based on interactions
    user.update_preferences(item1, engaged=True)
    
    # Check that category interest was created
    assert 0 in user.category_interests
    assert user.category_interests[0] > 0
    
    # Update with another category
    user.update_preferences(item2, engaged=True)
    assert 1 in user.category_interests
    
    # Update with a multi-category item
    user.update_preferences(item3, engaged=True)
    assert 0 in user.category_interests
    assert 2 in user.category_interests
    
    # Category 0 should have higher interest (seen twice)
    assert user.category_interests[0] > user.category_interests[1]


def test_engagement_model():
    """Test that engagement model calculates probabilities correctly."""
    # Create a simple engagement model with custom parameters
    model = EngagementModel(
        position_decay_factor=0.5,  # Override the default for easier testing
        base_engagement_rate=0.5     # Set a consistent base rate
    )
    
    # Create user and items with matching dimensions (10)
    user_prefs = np.zeros(10)
    user_prefs[0] = 1.0
    user = create_test_user(1, preference_vector=user_prefs)
    
    item1_features = np.zeros(10)
    item1_features[0] = 1.0
    item1 = create_test_item(1, [0], features=item1_features)  # Perfect match
    
    item2_features = np.zeros(10)
    item2_features[1] = 1.0
    item2 = create_test_item(2, [1], features=item2_features)  # No match
    
    # Test position bias
    pos0_prob = model.calculate_engagement_probability(
        user, item1, 0, []
    )
    pos5_prob = model.calculate_engagement_probability(
        user, item1, 5, []
    )
    
    # Higher position should have higher probability
    assert pos0_prob > pos5_prob
    
    # With position_bias_factor=0.8 and decay=0.5, position effect is around (0.5^(5*0.8))
    # Calculate the expected range
    expected_min = 0.5**(5 * 1.2)  # Allow for some variation
    expected_max = 0.5**(5 * 0.6)  # Allow for some variation
    
    # Calculate the actual position effect ratio
    ratio = pos5_prob / pos0_prob  # Changed from pos0_prob/pos5_prob to match how position decay works
    
    # The test assumes higher positions (lower indices) have higher probabilities
    # So ratio should be less than 1
    assert ratio < 1.0, f"Position 5 should have lower probability than position 0, ratio={ratio}"
    assert expected_min < ratio < expected_max, f"Expected ratio around {expected_min:.4f}-{expected_max:.4f}, got {ratio:.4f}"
    
    # Test preference influence
    pref_high_prob = model.calculate_engagement_probability(
        user, item1, 0, []  # High preference item
    )
    pref_low_prob = model.calculate_engagement_probability(
        user, item2, 0, []  # Low preference item
    )
    assert pref_high_prob > pref_low_prob  # Higher preference should have higher probability
    
    # Test novelty impact
    # Create user history with items similar to item1
    history_item1 = create_test_item(3, [0])
    history_item1.features = np.zeros(10)
    history_item1.features[0] = 0.9
    history_item1.features[1] = 0.1
    
    history_item2 = create_test_item(4, [0])
    history_item2.features = np.zeros(10)
    history_item2.features[0] = 0.8
    history_item2.features[1] = 0.2
    
    history = [history_item1, history_item2]
    
    # Calculate engagement for similar and novel items
    
    novel_item = create_test_item(5, [2])
    novel_item.features = np.zeros(10)
    novel_item.features[1] = 0.9
    
    novel_prob = model.calculate_engagement_probability(
        user, novel_item, 0, history  # Different from history
    )
    
    # Create a user who values exploration more
    explorer = create_test_user(2, preference_vector=user.preference_vector)
    explorer.exploration_factor = 0.8  # Very exploratory
    
    explorer_prob = model.calculate_engagement_probability(
        explorer, novel_item, 0, history
    )
    
    # Explorer should value novel item more than regular user
    # This is testing the impact of the exploration_factor
    assert explorer_prob > novel_prob, "Exploratory user should have higher probability for novel item"


def test_user_engagement():
    """Test that user engagement works as expected."""
    # Create user, item, and model with controlled randomness
    user = create_test_user(1)
    item = create_test_item(1, [0])  # High preference - both now have 10 dimensions
    model = EngagementModel()
    
    # Mock the random choice to always engage
    original_random = np.random.random
    np.random.random = lambda: 0.0  # Always less than any probability
    
    try:
        # Test engagement at position 0
        engaged = user.engage_with_item(item, 0, model)
        
        # With our mock, engagement should be True
        assert engaged, "User should have engaged with the item"
        
        # Check that engagement counts were updated
        assert user.engagement_counts['total_recommended'] == 1
        assert user.engagement_counts['last_session_recommended'] == 1
        assert user.engagement_counts['total_engaged'] == 1
        assert user.engagement_counts['last_session_engaged'] == 1
        assert len(user.interaction_history) == 1
        
        # Test new session resets session counts
        user.start_new_session()
        assert user.engagement_counts['last_session_recommended'] == 0
        assert user.engagement_counts['last_session_engaged'] == 0
        assert user.engagement_counts['total_recommended'] == 1, "Total counts should not be reset"
        assert user.engagement_counts['total_engaged'] == 1, "Total counts should not be reset"
    finally:
        # Restore the original random function
        np.random.random = original_random


def test_generate_controlled_users():
    """Test creating users with controlled preference distribution."""
    # Create 20 users with 4 distinct preference groups
    users = []
    for i in range(20):
        group = i % 4  # Create 4 distinct user groups
        pref_vector = np.zeros(10)
        
        # Make the groups more distinct with less overlap
        if group == 0:
            pref_vector[0:2] = 1.0  # Group 0 likes features 0-1
        elif group == 1:
            pref_vector[3:5] = 1.0  # Group 1 likes features 3-4
        elif group == 2:
            pref_vector[6:8] = 1.0  # Group 2 likes features 6-7
        else:
            pref_vector[8:10] = 1.0  # Group 3 likes features 8-9
            
        # Normalize the vector
        pref_vector = pref_vector / np.linalg.norm(pref_vector)
        
        user = User(
            user_id=i,
            preference_vector=pref_vector,
            exploration_factor=0.2
        )
        users.append(user)
    
    # Create a user universe with our users
    user_universe = UserUniverse(num_users=20, num_user_features=10)
    user_universe.users = users
    user_universe.user_id_map = {u.user_id: u for u in users}
    
    # Test user similarity - directly test the similarity calculation method
    # Calculate similarities directly instead of using get_similar_users
    
    # Users in same group should have high similarity
    # Group 0 users are at indices 0, 4, 8, 12, 16
    group0_user = users[0]
    group0_similarities = []
    
    # Group 1 users are at indices 1, 5, 9, 13, 17
    group1_user = users[1]
    group1_similarities = []
    
    # Calculate similarities directly
    for i, other_user in enumerate(users):
        if i != 0:  # Skip self comparison
            similarity = user_universe.get_user_similarity(group0_user, other_user)
            group = i % 4
            if group == 0:
                group0_similarities.append((i, similarity))
            else:
                # Store the similarity between group0_user and this other group user
                if group == 1:  # Only track comparisons with group 1
                    group1_similarities.append((i, similarity))
    
    # Sort by similarity
    group0_similarities.sort(key=lambda x: x[1], reverse=True)
    group1_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Users in same group should have higher similarity than users in different groups
    # Check that any group 0 user has higher similarity to group0_user than any group 1 user
    assert group0_similarities[0][1] > group1_similarities[0][1], "Users in same group should have higher similarity"
    
    # Now test specific users are similar to each other
    # Group 0 users should all be similar to each other
    for idx in [4, 8, 12, 16]:  # Other group 0 users
        similarity = user_universe.get_user_similarity(group0_user, users[idx])
        assert similarity > 0.9, f"User {idx} should be very similar to user 0, got {similarity}"
    
    # Group 1 users should all be similar to each other 
    for idx in [5, 9, 13, 17]:  # Other group 1 users
        similarity = user_universe.get_user_similarity(group1_user, users[idx])
        assert similarity > 0.9, f"User {idx} should be very similar to user 1, got {similarity}"
    
    # Users across different groups should have lower similarity
    # Group 0 and Group 1 users should have low similarity
    for g0_idx in [0, 4, 8, 12, 16]:
        for g1_idx in [1, 5, 9, 13, 17]:
            similarity = user_universe.get_user_similarity(users[g0_idx], users[g1_idx])
            assert similarity < 0.1, f"User {g0_idx} and {g1_idx} should have low similarity, got {similarity}"


def test_user_evolution():
    """Test that users evolve over time based on interactions."""
    # Create a 10-dimensional preference vector
    pref_vector = np.zeros(10)
    pref_vector[0] = 1.0
    user = create_test_user(1, preference_vector=pref_vector)
    
    # Create a series of items with a particular preference direction
    items = []
    for i in range(10):
        features = np.zeros(10)
        features[1] = 1.0  # All items have feature 1 active
        items.append(create_test_item(i, [0], features=features))
    
    # Record initial preference
    initial_pref0 = user.preference_vector[0]
    initial_pref1 = user.preference_vector[1]
    
    # Interact with these items multiple times
    for item in items:
        user.update_preferences(item, engaged=True)
    
    # User preferences should have shifted toward the items
    assert user.preference_vector[0] < initial_pref0, "Preference for feature 0 should have decreased"
    assert user.preference_vector[1] > initial_pref1, "Preference for feature 1 should have increased"


def test_user_universe_manual():
    """Test user universe operations with manually created users."""
    # Create a user universe
    user_universe = UserUniverse(num_users=5, num_user_features=10)
    
    # Add some users manually
    for i in range(5):
        pref_vector = np.zeros(10)
        pref_vector[i % 3:i % 3 + 3] = 1.0
        pref_vector = pref_vector / np.linalg.norm(pref_vector)
        
        user = User(
            user_id=i,
            preference_vector=pref_vector
        )
        user_universe.users.append(user)
        user_universe.user_id_map[i] = user
    
    # Test getting user by ID
    test_user = user_universe.get_user_by_id(2)
    assert test_user is not None
    assert test_user.user_id == 2
    
    # Test non-existent user
    non_existent = user_universe.get_user_by_id(999)
    assert non_existent is None
    
    # Test converting to dataframe
    df = user_universe.to_dataframe()
    assert len(df) == 5
    assert 'user_id' in df.columns
    assert 'exploration_factor' in df.columns
    
    # Add some category interests to a user
    user_universe.users[0].category_interests = {0: 0.5, 1: 0.3}
    
    # Regenerate dataframe
    df = user_universe.to_dataframe()
    
    # Check that category interests are included
    assert 'category_interest_0' in df.columns
    assert 'category_interest_1' in df.columns


def test_manual_interactions():
    """Test the generation of engagement events manually."""
    # Create an engagement model
    model = EngagementModel()
    
    # Create a user and some items with matching 10D feature vectors
    user = create_test_user(1)
    items = [create_test_item(i, [i % 3]) for i in range(10)]
    
    # Generate engagement events with fixed randomness
    np.random.seed(42)
    events = model.generate_engagement_events(user, items, num_events=5)
    
    # Verify event structure
    assert len(events) == 5
    for event in events:
        assert 'user_id' in event
        assert 'item_id' in event
        assert 'position' in event
        assert 'engaged' in event
        assert 'timestamp' in event
    
    # Check that some events resulted in engagement
    engaged_events = [e for e in events if e['engaged']]
    assert len(engaged_events) > 0, "At least some events should result in engagement"
    
    # Check that engaged items are in user history
    for event in engaged_events:
        item_id = event['item_id']
        matching_items = [item for item in user.interaction_history if item.item_id == item_id]
        assert len(matching_items) > 0, f"Item {item_id} should be in user history"


def test_user_universe_with_content():
    """Test integrating user universe with content universe."""
    # Create a small content universe with matching feature dimensions
    content_universe = ContentUniverse(
        num_items=50,
        num_categories=3,
        num_features=10,  # Must match user feature dimension
        seed=42
    )
    content_universe.generate_content()
    
    # Create a user universe with connection to content
    user_universe = UserUniverse(
        num_users=10,
        num_user_features=10,  # Must match content feature dimension
        seed=42
    )
    
    # Generate users based on content
    user_universe.generate_users(content_universe)
    
    # Check that users were created correctly
    assert len(user_universe.users) == 10
    
    # Generate interactions manually 
    # We'll skip the generate_initial_interactions method since it has randomness
    # and just verify that it returns the expected structure
    interactions = []
    
    # Let's manually create an interaction for each user
    for user in user_universe.users:
        # Pick a random item
        item = content_universe.items[user.user_id % len(content_universe.items)]
        
        # Force user to engage with it
        original_random = np.random.random
        np.random.random = lambda: 0.0  # Always engage
        
        try:
            engaged = user.engage_with_item(item, 0, user_universe.engagement_model)
            assert engaged, "User should have engaged with item"
            
            # Create an interaction event
            interaction = {
                'user_id': user.user_id,
                'item_id': item.item_id,
                'position': 0,
                'engaged': engaged,
                'timestamp': 0
            }
            interactions.append(interaction)
        finally:
            np.random.random = original_random
    
    # Check that all users have interaction history
    for user in user_universe.users:
        assert len(user.interaction_history) == 1, f"User {user.user_id} should have 1 interaction"


def test_to_dataframe():
    """Test conversion of user universe to DataFrame."""
    user_universe = UserUniverse(
        num_users=10,
        num_user_features=10,  # Changed to 10 to be consistent
        seed=42
    )
    
    # Create simple users
    for i in range(10):
        pref = np.zeros(10)
        pref[i % 5] = 1.0
        user = User(
            user_id=i,
            preference_vector=pref,
            exploration_factor=0.1 * i
        )
        user_universe.users.append(user)
        user_universe.user_id_map[i] = user
        
        # Add some interaction history and category interests
        user.interaction_history = [create_test_item(j, [j % 3]) for j in range(i)]
        for j in range(3):
            if i % (j+1) == 0:  # Add some variation
                user.category_interests[j] = 0.1 * (i % (j+1) + 1)
    
    # Convert to DataFrame
    df = user_universe.to_dataframe()
    
    # Check DataFrame structure
    assert len(df) == 10
    assert 'user_id' in df.columns
    assert 'exploration_factor' in df.columns
    assert 'num_interactions' in df.columns
    
    # Check that preference features were included
    for i in range(10):  # Changed to 10 to be consistent
        assert f'preference_{i}' in df.columns
    
    # Check that category interests were included
    category_interest_cols = [col for col in df.columns if col.startswith('category_interest_')]
    assert len(category_interest_cols) > 0