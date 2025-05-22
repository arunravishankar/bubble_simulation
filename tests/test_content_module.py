import pytest
import numpy as np
from bubble_simulation.content import Item, ContentUniverse


def test_item_similarity():
    """Test that item similarity calculation works correctly."""
    # Create two items with known feature vectors
    item1 = Item(
        item_id=1,
        features=np.array([1, 0, 0]),
        categories=[0],
        popularity_score=0.5
    )
    
    item2 = Item(
        item_id=2,
        features=np.array([0, 1, 0]),
        categories=[0],
        popularity_score=0.3
    )
    
    item3 = Item(
        item_id=3,
        features=np.array([1, 1, 0]),
        categories=[0, 1],
        popularity_score=0.2
    )
    
    # Test similarity calculations
    assert item1.similarity(item1) == pytest.approx(1.0)  # Same item
    assert item1.similarity(item2) == pytest.approx(0.0)  # Orthogonal
    assert item1.similarity(item3) == pytest.approx(1.0 / np.sqrt(2))  # 45 degrees
    
    # Test with zero vectors
    item_zero = Item(
        item_id=4,
        features=np.array([0, 0, 0]),
        categories=[0],
        popularity_score=0.1
    )
    assert item1.similarity(item_zero) == 0.0
    assert item_zero.similarity(item_zero) == 0.0


def test_content_universe_generation():
    """Test that content universe generation works as expected."""
    # Create a small universe for testing
    universe = ContentUniverse(
        num_items=100,
        num_categories=5,
        num_features=10,
        seed=42
    )
    
    # Generate content
    universe.generate_content()
    
    # Check that the correct number of items were created
    assert len(universe.items) == 100
    
    # Check that all items have the correct feature dimensionality
    for item in universe.items:
        assert len(item.features) == 10
        
    # Check that item IDs are unique
    item_ids = [item.item_id for item in universe.items]
    assert len(item_ids) == len(set(item_ids))
    
    # Check that all items are assigned to at least one valid category
    for item in universe.items:
        assert len(item.categories) > 0
        assert all(0 <= cat < 5 for cat in item.categories)
        
    # Check that category_items contains all items
    all_category_items = set()
    for items in universe.category_items.values():
        all_category_items.update(items)
    assert len(all_category_items) == 100
    
    # Check multi-category assignment
    items_with_multiple_categories = [item for item in universe.items if len(item.categories) > 1]
    assert len(items_with_multiple_categories) > 0, "No items have multiple categories"
    
    # Check popularity distribution
    popularity_scores = universe.get_popularity_distribution()
    assert len(popularity_scores) == 100
    assert all(0 <= score <= 1 for score in popularity_scores)


def test_dynamic_content():
    """Test adding new content over time."""
    universe = ContentUniverse(
        num_items=100,
        num_categories=5,
        num_features=10,
        dynamic_content=True,
        content_growth_rate=0.1,
        seed=42
    )
    
    # Generate initial content
    universe.generate_content()
    initial_count = len(universe.items)
    assert initial_count == 100
    
    # Add new content at timestep 1
    new_items = universe.add_new_content(timestep=1)
    assert len(new_items) == 10  # 10% growth rate
    assert len(universe.items) == 110
    
    # Check that new items have the correct creation time
    for item in new_items:
        assert item.creation_time == 1
        
    # Get new content since timestep 0
    new_content = universe.get_new_content_since(0)
    assert len(new_content) == 10
    
    # Add more content at timestep 2
    more_new_items = universe.add_new_content(timestep=2)
    assert len(more_new_items) == 11  # 10% of 110
    assert len(universe.items) == 121
    
    # Check that the new content is retrievable by ID
    for item in more_new_items:
        retrieved_item = universe.get_item_by_id(item.item_id)
        assert retrieved_item is not None
        assert retrieved_item.item_id == item.item_id
        
    # Check multi-category assignment in new items
    items_with_multiple_categories = [item for item in new_items if len(item.categories) > 1]
    assert len(items_with_multiple_categories) > 0, "No new items have multiple categories"


def test_get_similar_items():
    """Test finding similar items."""
    universe = ContentUniverse(
        num_items=100,
        num_categories=5,
        num_features=10,
        seed=42
    )
    
    # Generate content
    universe.generate_content()
    
    # Pick a random item
    item = universe.items[0]
    
    # Get similar items
    similar_items = universe.get_similar_items(item, n=5)
    
    # Check that we got the right number
    assert len(similar_items) == 5
    
    # Check that the item itself is not in the results
    assert item not in similar_items
    
    # Check that items are sorted by similarity
    similarities = [item.similarity(other) for other in similar_items]
    assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))
    
    # Test with exclude_ids
    exclude_ids = {similar_items[0].item_id, similar_items[1].item_id}
    filtered_similar_items = universe.get_similar_items(item, n=5, exclude_ids=exclude_ids)
    
    # Check that excluded items are not in results
    for similar_item in filtered_similar_items:
        assert similar_item.item_id not in exclude_ids


def test_multi_category_similar_items():
    """Test finding similar items across multiple categories."""
    universe = ContentUniverse(
        num_items=100,
        num_categories=5,
        num_features=10,
        seed=42
    )
    
    # Generate content
    universe.generate_content()
    
    # Find an item with multiple categories
    multi_cat_items = [item for item in universe.items if len(item.categories) > 1]
    assert len(multi_cat_items) > 0, "No items with multiple categories found"
    
    test_item = multi_cat_items[0]
    
    # Get similar items
    similar_items = universe.get_similar_items(test_item, n=10)
    
    # Check that we got items from multiple categories
    categories_in_results = set()
    for item in similar_items:
        categories_in_results.update(item.categories)
        
    assert len(categories_in_results) > 1, "Similar items only from one category"
    
    # Check that some items share categories with test item
    shared_category_items = [item for item in similar_items 
                             if any(cat in test_item.categories for cat in item.categories)]
    assert len(shared_category_items) > 0, "No similar items share categories with test item"


def test_to_dataframe():
    """Test conversion to DataFrame."""
    universe = ContentUniverse(
        num_items=100,
        num_categories=5,
        num_features=3,
        seed=42
    )
    
    # Generate content
    universe.generate_content()
    
    # Add some metadata
    universe.items[0].metadata['test_key'] = 'test_value'
    
    # Convert to DataFrame
    df = universe.to_dataframe()
    
    # Check DataFrame structure
    assert len(df) == 100
    assert 'item_id' in df.columns
    assert 'primary_category' in df.columns
    assert 'categories' in df.columns
    assert 'num_categories' in df.columns
    assert 'popularity_score' in df.columns
    assert 'feature_0' in df.columns
    assert 'feature_1' in df.columns
    assert 'feature_2' in df.columns
    assert 'metadata_test_key' in df.columns
    
    # Check that multi-category information is preserved
    assert df['num_categories'].max() > 1, "No items with multiple categories"
    assert isinstance(df['categories'].iloc[0], list), "Categories not stored as lists"
