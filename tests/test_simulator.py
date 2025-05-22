import numpy as np
import pandas as pd
from bubble_simulation.simulation.simulator import Simulator, SimulationResults, run_comparative_simulation
from bubble_simulation.content.content_universe import ContentUniverse
from bubble_simulation.users.user_universe import UserUniverse
from bubble_simulation.recommenders.popularity import PopularityRecommender
from bubble_simulation.recommenders.collaborative import CollaborativeFilteringRecommender
from bubble_simulation.recommenders.reinforcement import RLRecommender


# Helper functions to create test data
def create_test_content_universe(num_items=20, num_categories=3, seed=42):
    """Create a small content universe for testing."""
    content_universe = ContentUniverse(
        num_items=num_items,
        num_categories=num_categories,
        num_features=10,
        seed=seed
    )
    content_universe.generate_content()
    return content_universe


def create_test_user_universe(num_users=10, num_features=10, seed=42):
    """Create a small user universe for testing."""
    user_universe = UserUniverse(
        num_users=num_users,
        num_user_features=num_features,
        seed=seed
    )
    return user_universe


def create_test_simulator(num_timesteps=5, recommendations_per_step=3, seed=42):
    """Create a test simulator with basic components."""
    content_universe = create_test_content_universe(seed=seed)
    user_universe = create_test_user_universe(seed=seed)
    user_universe.generate_users(content_universe)
    
    recommender = PopularityRecommender(retrain_frequency=2, top_k=5)
    
    simulator = Simulator(
        content_universe=content_universe,
        user_universe=user_universe,
        recommender=recommender,
        num_timesteps=num_timesteps,
        recommendations_per_step=recommendations_per_step,
        seed=seed
    )
    
    # Fix: Initialize user_shown_items properly
    for user in user_universe.users:
        simulator.user_shown_items[user.user_id] = set()
    
    return simulator

# Test SimulationResults class
def test_simulation_results_initialization():
    """Test that SimulationResults initializes correctly."""
    results = SimulationResults(
        recommender_name="TestRecommender",
        num_timesteps=10,
        num_users=5,
        num_items=20
    )
    
    assert results.recommender_name == "TestRecommender"
    assert results.num_timesteps == 10
    assert results.num_users == 5
    assert results.num_items == 20
    
    # Check that lists are initialized as empty
    assert results.timesteps == []
    assert results.total_interactions == []
    assert results.content_diversity == []
    assert results.recommender_metrics == {}
    assert results.interaction_history == []


def test_simulation_results_to_dataframe():
    """Test converting SimulationResults to DataFrame."""
    results = SimulationResults(
        recommender_name="TestRecommender",
        num_timesteps=3,
        num_users=5,
        num_items=20
    )
    
    # Add some test data
    results.timesteps = [0, 1, 2]
    results.total_interactions = [15, 18, 20]
    results.total_engagements = [8, 10, 12]
    results.engagement_rate = [0.53, 0.56, 0.60]
    results.content_diversity = [0.3, 0.4, 0.5]
    results.category_diversity = [0.7, 0.8, 0.9]
    results.user_content_diversity = [0.2, 0.3, 0.4]
    results.average_ctr = [0.53, 0.56, 0.60]
    results.catalog_coverage = [0.3, 0.4, 0.5]
    results.avg_user_engagement_rate = [0.5, 0.55, 0.58]
    results.avg_user_diversity = [0.25, 0.35, 0.45]
    
    # Add recommender metrics
    results.recommender_metrics = {
        'train_time': [0.1, 0.15, 0.12],
        'epsilon': [0.1, 0.09, 0.08]
    }
    
    # Convert to DataFrame
    df = results.to_dataframe()
    
    # Check DataFrame structure
    assert len(df) == 3
    assert 'timestep' in df.columns
    assert 'total_interactions' in df.columns
    assert 'content_diversity' in df.columns
    assert 'recommender_train_time' in df.columns
    assert 'recommender_epsilon' in df.columns
    
    # Check values
    assert df['timestep'].tolist() == [0, 1, 2]
    assert df['total_interactions'].tolist() == [15, 18, 20]
    assert df['content_diversity'].tolist() == [0.3, 0.4, 0.5]


# Test Simulator class
def test_simulator_initialization():
    """Test that Simulator initializes correctly."""
    content_universe = create_test_content_universe()
    user_universe = create_test_user_universe()
    user_universe.generate_users(content_universe)
    recommender = PopularityRecommender()
    
    simulator = Simulator(
        content_universe=content_universe,
        user_universe=user_universe,
        recommender=recommender,
        num_timesteps=10,
        recommendations_per_step=5,
        seed=42
    )
    
    assert simulator.content_universe == content_universe
    assert simulator.user_universe == user_universe
    assert simulator.recommender == recommender
    assert simulator.num_timesteps == 10
    assert simulator.recommendations_per_step == 5
    assert simulator.current_timestep == 0
    assert simulator.all_interactions == []
    assert simulator.timestep_interactions == []
    assert simulator.shown_items_global == set()
    assert len(simulator.user_shown_items) == len(user_universe.users)


def test_simulator_reset():
    """Test that simulator can be reset."""
    simulator = create_test_simulator()
    
    # Modify some state
    simulator.current_timestep = 5
    simulator.all_interactions = [{'test': 'data'}]
    simulator.shown_items_global = {1, 2, 3}
    simulator.user_shown_items[0] = {1, 2}
    
    # Reset
    simulator.reset_simulation()
    
    # Check that state was reset
    assert simulator.current_timestep == 0
    assert simulator.all_interactions == []
    assert simulator.timestep_interactions == []
    assert simulator.shown_items_global == set()
    assert all(len(items) == 0 for items in simulator.user_shown_items.values())
    
    # Check that users were reset
    for user in simulator.user_universe.users:
        assert user.interaction_history == []
        assert user.category_interests == {}
        assert user.engagement_counts['total_recommended'] == 0
        assert user.engagement_counts['total_engaged'] == 0


def test_run_single_timestep():
    """Test running a single timestep."""
    simulator = create_test_simulator(num_timesteps=1)
    
    # Run one timestep manually
    metrics = simulator._run_timestep()
    
    # Check that metrics were returned
    assert isinstance(metrics, dict)
    assert 'total_interactions' in metrics
    assert 'total_engagements' in metrics
    assert 'engagement_rate' in metrics
    assert 'unique_items_shown' in metrics
    assert 'unique_users_active' in metrics
    
    # Check that interactions were created
    assert len(simulator.timestep_interactions) > 0
    assert len(simulator.all_interactions) > 0
    
    # Check interaction structure
    for interaction in simulator.timestep_interactions:
        assert 'timestep' in interaction
        assert 'user_id' in interaction
        assert 'item_id' in interaction
        assert 'position' in interaction
        assert 'engaged' in interaction
        assert 'timestamp' in interaction
        assert isinstance(interaction['engaged'], (bool, np.bool_))


def test_diversity_calculations():
    """Test diversity calculation methods."""
    simulator = create_test_simulator()
    
    # Add some test interactions
    simulator.timestep_interactions = [
        {'item_id': 0, 'user_id': 0},
        {'item_id': 1, 'user_id': 1},
        {'item_id': 2, 'user_id': 2},
    ]
    
    # Add some shown items
    simulator.shown_items_global = {0, 1, 2, 3, 4}
    simulator.user_shown_items = {
        0: {0, 1},
        1: {1, 2, 3},
        2: {2, 4}
    }
    
    # Test content diversity
    content_diversity = simulator._calculate_content_diversity()
    expected_diversity = len(simulator.shown_items_global) / len(simulator.content_universe.items)
    assert content_diversity == expected_diversity
    
    # Test category diversity
    category_diversity = simulator._calculate_category_diversity()
    assert 0 <= category_diversity <= 1
    
    # Test user content diversity
    user_diversity = simulator._calculate_user_content_diversity()
    assert 0 <= user_diversity <= 1


def test_metrics_recording():
    """Test that metrics are recorded correctly."""
    simulator = create_test_simulator()
    
    # Create mock results
    results = SimulationResults(
        recommender_name="TestRecommender",
        num_timesteps=5,
        num_users=10,
        num_items=20
    )
    
    # Create mock timestep metrics
    timestep_metrics = {
        'total_interactions': 25,
        'total_engagements': 15,
        'engagement_rate': 0.6,
        'unique_items_shown': 8,
        'unique_users_active': 10
    }
    
    # Record metrics
    simulator._record_metrics(results, timestep_metrics)
    
    # Check that basic metrics were recorded
    assert len(results.timesteps) == 1
    assert results.timesteps[0] == 0
    assert results.total_interactions[0] == 25
    assert results.total_engagements[0] == 15
    assert results.engagement_rate[0] == 0.6
    
    # Check that diversity metrics were calculated and recorded
    assert len(results.content_diversity) == 1
    assert len(results.category_diversity) == 1
    assert len(results.user_content_diversity) == 1
    assert 0 <= results.content_diversity[0] <= 1
    assert 0 <= results.category_diversity[0] <= 1
    assert 0 <= results.user_content_diversity[0] <= 1


def test_full_simulation():
    """Test running a complete simulation."""
    simulator = create_test_simulator(num_timesteps=3, recommendations_per_step=2)
    
    # Run full simulation
    results = simulator.run_simulation()
    
    # Check results structure
    assert isinstance(results, SimulationResults)
    assert results.recommender_name == simulator.recommender.name
    assert results.num_timesteps == 3
    assert results.num_users == len(simulator.user_universe.users)
    assert results.num_items == len(simulator.content_universe.items)
    
    # Check that time series data has correct length
    assert len(results.timesteps) == 3
    assert len(results.total_interactions) == 3
    assert len(results.engagement_rate) == 3
    assert len(results.content_diversity) == 3
    
    # Check that timesteps are sequential
    assert results.timesteps == [0, 1, 2]
    
    # Check that interactions were created
    assert len(simulator.all_interactions) > 0
    
    # Check that some items were shown
    assert len(simulator.shown_items_global) > 0
    
    # Check that engagement rates are reasonable
    for rate in results.engagement_rate:
        assert 0 <= rate <= 1


def test_simulation_with_different_recommenders():
    """Test simulation with different types of recommenders."""
    content_universe = create_test_content_universe()
    user_universe = create_test_user_universe()
    user_universe.generate_users(content_universe)
    
    # Test with PopularityRecommender
    pop_recommender = PopularityRecommender(top_k=3)
    pop_simulator = Simulator(
        content_universe=content_universe,
        user_universe=user_universe,
        recommender=pop_recommender,
        num_timesteps=2,
        recommendations_per_step=2
    )
    
    pop_results = pop_simulator.run_simulation()
    assert pop_results.recommender_name == "PopularityRecommender"
    assert len(pop_results.timesteps) == 2
    
    # Reset for next test
    pop_simulator.reset_simulation()
    
    # Test with CollaborativeFilteringRecommender
    cf_recommender = CollaborativeFilteringRecommender(top_k=3, num_factors=5)
    cf_simulator = Simulator(
        content_universe=content_universe,
        user_universe=user_universe,
        recommender=cf_recommender,
        num_timesteps=2,
        recommendations_per_step=2
    )
    
    cf_results = cf_simulator.run_simulation()
    assert cf_results.recommender_name == "CollaborativeFilteringRecommender"
    assert len(cf_results.timesteps) == 2


def test_get_interaction_dataframe():
    """Test converting interactions to DataFrame."""
    simulator = create_test_simulator(num_timesteps=2)
    
    # Run simulation to generate interactions
    simulator.run_simulation()
    
    # Get interaction DataFrame
    df = simulator.get_interaction_dataframe()
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    
    # Check required columns
    required_columns = [
        'timestep', 'user_id', 'item_id', 'position', 'engaged', 'timestamp',
        'item_categories', 'item_popularity', 'item_creation_time',
        'user_exploration_factor', 'user_position_bias_factor', 'user_diversity_preference'
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check data types
    assert df['timestep'].dtype in [np.int64, int]
    assert df['user_id'].dtype in [np.int64, int]
    assert df['item_id'].dtype in [np.int64, int]
    assert df['engaged'].dtype == bool


def test_empty_simulation():
    """Test simulation with edge cases."""
    # Test with very small numbers
    content_universe = create_test_content_universe(num_items=1, num_categories=1)
    user_universe = create_test_user_universe(num_users=1)
    user_universe.generate_users(content_universe)
    
    recommender = PopularityRecommender(top_k=1)
    
    simulator = Simulator(
        content_universe=content_universe,
        user_universe=user_universe,
        recommender=recommender,
        num_timesteps=1,
        recommendations_per_step=1
    )
    
    # Should not crash
    results = simulator.run_simulation()
    assert isinstance(results, SimulationResults)
    assert len(results.timesteps) == 1


def test_comparative_simulation():
    """Test running comparative simulations with multiple recommenders."""
    content_universe = create_test_content_universe()
    user_universe = create_test_user_universe()
    user_universe.generate_users(content_universe)
    
    # Create different recommenders
    recommenders = [
        PopularityRecommender(name="Popularity", top_k=3),
        CollaborativeFilteringRecommender(name="CollabFilter", top_k=3, num_factors=5),
        RLRecommender(name="RL", top_k=3, batch_size=2, memory_size=10)
    ]
    
    # Run comparative simulation
    results = run_comparative_simulation(
        content_universe=content_universe,
        user_universe=user_universe,
        recommenders=recommenders,
        num_timesteps=2,
        recommendations_per_step=2,
        seed=42
    )
    
    # Check results
    assert isinstance(results, dict)
    assert len(results) == 3
    assert "Popularity" in results
    assert "CollabFilter" in results
    assert "RL" in results
    
    # Check that each result is valid
    for name, result in results.items():
        assert isinstance(result, SimulationResults)
        assert result.recommender_name == name
        assert len(result.timesteps) == 2
        assert len(result.total_interactions) == 2


def test_simulation_with_dynamic_content():
    """Test simulation with dynamic content addition."""
    content_universe = create_test_content_universe()
    content_universe.dynamic_content = True
    content_universe.content_growth_rate = 0.1
    
    user_universe = create_test_user_universe()
    user_universe.generate_users(content_universe)
    
    recommender = PopularityRecommender(top_k=3)
    
    simulator = Simulator(
        content_universe=content_universe,
        user_universe=user_universe,
        recommender=recommender,
        num_timesteps=3,
        recommendations_per_step=2
    )
    
    initial_item_count = len(content_universe.items)
    
    # Run simulation
    results = simulator.run_simulation()
    
    # Check that new items were added
    final_item_count = len(content_universe.items)
    assert final_item_count >= initial_item_count
    
    # Simulation should still complete successfully
    assert len(results.timesteps) == 3


def test_simulation_metrics_consistency():
    """Test that simulation metrics are consistent and reasonable."""
    simulator = create_test_simulator(num_timesteps=5)
    results = simulator.run_simulation()
    
    # Check that diversity metrics are between 0 and 1
    for diversity in results.content_diversity:
        assert 0 <= diversity <= 1
    
    for diversity in results.category_diversity:
        assert 0 <= diversity <= 1
    
    for diversity in results.user_content_diversity:
        assert 0 <= diversity <= 1
    
    # Check that engagement rates are between 0 and 1
    for rate in results.engagement_rate:
        assert 0 <= rate <= 1
    
    for rate in results.avg_user_engagement_rate:
        assert 0 <= rate <= 1
    
    # Check that catalog coverage is between 0 and 1
    for coverage in results.catalog_coverage:
        assert 0 <= coverage <= 1
    
    # Check that interactions and engagements are non-negative
    for interactions in results.total_interactions:
        assert interactions >= 0
    
    for engagements in results.total_engagements:
        assert engagements >= 0
    
    # Check that engagements <= interactions
    for i in range(len(results.total_interactions)):
        assert results.total_engagements[i] <= results.total_interactions[i]


def test_recommender_metrics_integration():
    """Test that recommender-specific metrics are properly integrated."""
    # Use RL recommender which has many metrics
    content_universe = create_test_content_universe()
    user_universe = create_test_user_universe()
    user_universe.generate_users(content_universe)
    
    recommender = RLRecommender(top_k=3, batch_size=2, memory_size=10)
    
    simulator = Simulator(
        content_universe=content_universe,
        user_universe=user_universe,
        recommender=recommender,
        num_timesteps=3,
        recommendations_per_step=2
    )
    
    results = simulator.run_simulation()
    
    # Check that recommender metrics were captured
    assert len(results.recommender_metrics) > 0
    
    # Check that metrics have correct length
    for metric_name, values in results.recommender_metrics.items():
        assert len(values) == len(results.timesteps), f"Metric {metric_name} has wrong length"
    
    # Convert to DataFrame and check
    df = results.to_dataframe()
    
    # Should have recommender metric columns
    recommender_columns = [col for col in df.columns if col.startswith('recommender_')]
    assert len(recommender_columns) > 0