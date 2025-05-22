import pytest
import numpy as np
import pandas as pd
from bubble_simulation.simulation.metrics import MetricsCalculator, FilterBubbleAnalyzer
from bubble_simulation.simulation.simulator import SimulationResults


# Helper functions to create test data
def create_test_simulation_results(recommender_name="TestRecommender", 
                                 num_timesteps=10) -> SimulationResults:
    """Create test simulation results with realistic data."""
    results = SimulationResults(
        recommender_name=recommender_name,
        num_timesteps=num_timesteps,
        num_users=50,
        num_items=100
    )
    
    # Add realistic time series data
    results.timesteps = list(range(num_timesteps))
    results.total_interactions = [100 + i * 5 for i in range(num_timesteps)]
    results.total_engagements = [50 + i * 2 for i in range(num_timesteps)]
    results.engagement_rate = [eng / inter for eng, inter in zip(results.total_engagements, results.total_interactions)]
    
    # Simulate declining diversity (filter bubble formation)
    results.content_diversity = [0.8 - i * 0.05 for i in range(num_timesteps)]
    results.category_diversity = [0.7 - i * 0.03 for i in range(num_timesteps)]
    results.user_content_diversity = [0.6 - i * 0.04 for i in range(num_timesteps)]
    
    # Business metrics
    results.catalog_coverage = [0.5 + i * 0.02 for i in range(num_timesteps)]
    results.average_ctr = results.engagement_rate.copy()
    results.avg_user_engagement_rate = [rate * 0.9 for rate in results.engagement_rate]
    results.avg_user_diversity = [div * 0.8 for div in results.user_content_diversity]
    
    # Recommender metrics
    results.recommender_metrics = {
        'train_time': [0.1 + i * 0.01 for i in range(num_timesteps)],
        'epsilon': [0.1 - i * 0.005 for i in range(num_timesteps)]
    }
    
    return results

def create_test_interactions_df(num_interactions=1000, num_users=50, num_items=100, num_timesteps=10) -> pd.DataFrame:
    """Create test interactions DataFrame with realistic patterns."""
    np.random.seed(42)  # For reproducible tests
    
    interactions = []
    
    for i in range(num_interactions):
        timestep = np.random.randint(0, num_timesteps)
        user_id = np.random.randint(0, num_users)
        
        # Simulate filter bubble: users tend to interact with fewer items over time
        if timestep < 3:
            # Early timesteps: more diverse
            item_id = np.random.randint(0, num_items)
        else:
            # Later timesteps: more concentrated
            item_id = np.random.randint(0, min(50, num_items))
        
        position = np.random.randint(0, 10)
        engaged = np.random.random() > (0.3 + position * 0.05)  # Position bias
        
        # Item characteristics
        item_popularity = np.random.beta(2, 5)  # Skewed toward low popularity
        item_categories = [np.random.randint(0, 10) for _ in range(np.random.randint(1, 4))]
        item_creation_time = np.random.randint(0, timestep + 1)
        
        # User characteristics
        user_exploration_factor = np.random.uniform(0.1, 0.9)
        user_position_bias_factor = np.random.uniform(0.5, 0.95)
        user_diversity_preference = np.random.uniform(0.2, 0.8)
        
        interaction = {
            'timestep': timestep,
            'user_id': user_id,
            'item_id': item_id,
            'position': position,
            'engaged': engaged,
            'timestamp': timestep * 100 + i,
            'item_popularity': item_popularity,
            'item_categories': item_categories,
            'item_creation_time': item_creation_time,
            'user_exploration_factor': user_exploration_factor,
            'user_position_bias_factor': user_position_bias_factor,
            'user_diversity_preference': user_diversity_preference
        }
        
        interactions.append(interaction)
    
    return pd.DataFrame(interactions)


# Test MetricsCalculator class
def test_metrics_calculator_initialization():
    """Test that MetricsCalculator initializes correctly."""
    calculator = MetricsCalculator()
    assert calculator is not None


def test_calculate_diversity_metrics():
    """Test diversity metrics calculation."""
    calculator = MetricsCalculator()
    results = create_test_simulation_results()
    interactions_df = create_test_interactions_df()
    
    diversity_metrics = calculator.calculate_diversity_metrics(results, interactions_df)
    
    # Check that all expected metrics are present
    expected_keys = [
        'content_diversity_trend', 'content_diversity_final', 'content_diversity_change',
        'category_diversity_trend', 'category_diversity_final', 'category_diversity_change',
        'user_diversity_trend', 'user_diversity_final', 'user_diversity_change',
        'bubble_formation_rate', 'diversity_gini_coefficient', 'long_tail_exposure'
    ]
    
    for key in expected_keys:
        assert key in diversity_metrics, f"Missing key: {key}"
    
    # Check that trends are correctly identified
    assert diversity_metrics['content_diversity_trend'] == 'decreasing'  # Simulated declining diversity
    
    # Check that final values are reasonable
    assert 0 <= diversity_metrics['content_diversity_final'] <= 1
    assert 0 <= diversity_metrics['diversity_gini_coefficient'] <= 1
    assert 0 <= diversity_metrics['long_tail_exposure'] <= 1


def test_calculate_engagement_metrics():
    """Test engagement metrics calculation."""
    calculator = MetricsCalculator()
    results = create_test_simulation_results()
    interactions_df = create_test_interactions_df()
    
    engagement_metrics = calculator.calculate_engagement_metrics(results, interactions_df)
    
    # Check that all expected metrics are present
    expected_keys = [
        'engagement_rate_trend', 'engagement_rate_final', 'engagement_rate_change',
        'user_engagement_trend', 'user_engagement_variance',
        'position_bias_impact', 'engagement_by_novelty', 'engagement_by_popularity',
        'session_length_trend', 'repeat_engagement_rate'
    ]
    
    for key in expected_keys:
        assert key in engagement_metrics, f"Missing key: {key}"
    
    # Check value ranges
    assert 0 <= engagement_metrics['engagement_rate_final'] <= 1
    assert engagement_metrics['user_engagement_variance'] >= 0
    assert 0 <= engagement_metrics['position_bias_impact'] <= 1
    assert 0 <= engagement_metrics['repeat_engagement_rate'] <= 1


def test_calculate_business_metrics():
    """Test business metrics calculation."""
    calculator = MetricsCalculator()
    results = create_test_simulation_results()
    interactions_df = create_test_interactions_df()
    
    business_metrics = calculator.calculate_business_metrics(results, interactions_df)
    
    # Check that all expected metrics are present
    expected_keys = [
        'catalog_coverage_trend', 'catalog_coverage_final', 'catalog_utilization_efficiency',
        'estimated_revenue_trend', 'revenue_per_user',
        'user_retention_rate', 'user_churn_indicators',
        'content_turnover_rate', 'niche_content_performance'
    ]
    
    for key in expected_keys:
        assert key in business_metrics, f"Missing key: {key}"
    
    # Check value ranges
    assert 0 <= business_metrics['catalog_coverage_final'] <= 1
    assert business_metrics['catalog_utilization_efficiency'] >= 0
    assert business_metrics['revenue_per_user'] >= 0
    assert 0 <= business_metrics['user_retention_rate'] <= 1
    assert 0 <= business_metrics['niche_content_performance'] <= 1


def test_calculate_bubble_formation_metrics():
    """Test bubble formation metrics calculation."""
    calculator = MetricsCalculator()
    results = create_test_simulation_results()
    interactions_df = create_test_interactions_df()
    
    bubble_metrics = calculator.calculate_bubble_formation_metrics(results, interactions_df)
    
    # Check that all expected metrics are present
    expected_keys = [
        'bubble_formation_velocity', 'content_concentration_trend',
        'interest_narrowing_rate', 'serendipity_trend',
        'echo_chamber_strength', 'novelty_decay_rate'
    ]
    
    for key in expected_keys:
        assert key in bubble_metrics, f"Missing key: {key}"
    
    # Check that bubble formation velocity is positive (diversity is declining)
    assert bubble_metrics['bubble_formation_velocity'] >= 0
    
    # Check that serendipity trend is a list
    assert isinstance(bubble_metrics['serendipity_trend'], list)
    
    # Check value ranges
    assert bubble_metrics['interest_narrowing_rate'] >= 0
    assert 0 <= bubble_metrics['echo_chamber_strength'] <= 1


def test_calculate_comparative_metrics():
    """Test comparative metrics calculation."""
    calculator = MetricsCalculator()
    
    # Create multiple simulation results
    results_dict = {
        'Traditional': create_test_simulation_results('Traditional'),
        'RL': create_test_simulation_results('RL')
    }
    
    # Create different interaction patterns
    interactions_dict = {
        'Traditional': create_test_interactions_df(500, 25, 50, 10),  # More concentrated
        'RL': create_test_interactions_df(500, 25, 80, 10)  # More diverse
    }
    
    comparative_metrics = calculator.calculate_comparative_metrics(results_dict, interactions_dict)
    
    # Check that all expected sections are present
    expected_keys = [
        'diversity_comparison', 'engagement_comparison', 
        'business_comparison', 'statistical_tests'
    ]
    
    for key in expected_keys:
        assert key in comparative_metrics, f"Missing key: {key}"
    
    # Check that both recommenders are compared
    assert 'Traditional' in comparative_metrics['diversity_comparison']
    assert 'RL' in comparative_metrics['diversity_comparison']
    
    # Check statistical tests structure
    assert 'diversity_tests' in comparative_metrics['statistical_tests']
    assert 'engagement_tests' in comparative_metrics['statistical_tests']


def test_helper_methods():
    """Test helper methods in MetricsCalculator."""
    calculator = MetricsCalculator()
    
    # Test diversity trend calculation
    increasing_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    decreasing_values = [0.7, 0.6, 0.5, 0.4, 0.3]
    stable_values = [0.5, 0.51, 0.49, 0.5, 0.52]
    
    assert calculator._calculate_diversity_trend(increasing_values) == 'increasing'
    assert calculator._calculate_diversity_trend(decreasing_values) == 'decreasing'
    assert calculator._calculate_diversity_trend(stable_values) == 'stable'
    
    # Test metric change calculation
    change = calculator._calculate_metric_change([0.5, 0.6])
    assert change == pytest.approx(20.0)  # 20% increase
    
    change = calculator._calculate_metric_change([0.6, 0.5])
    assert change == pytest.approx(-16.67, rel=1e-2)  # ~16.67% decrease
    
    # Test bubble formation rate
    bubble_rate = calculator._calculate_bubble_formation_rate(decreasing_values)
    assert bubble_rate > 0  # Should detect declining trend


def test_edge_cases():
    """Test edge cases and empty data handling."""
    calculator = MetricsCalculator()
    
    # Empty results
    empty_results = SimulationResults(
        recommender_name="Empty",
        num_timesteps=0,
        num_users=0,
        num_items=0
    )
    
    empty_df = pd.DataFrame()
    
    # Should not crash with empty data
    diversity_metrics = calculator.calculate_diversity_metrics(empty_results, empty_df)
    assert isinstance(diversity_metrics, dict)
    
    engagement_metrics = calculator.calculate_engagement_metrics(empty_results, empty_df)
    assert isinstance(engagement_metrics, dict)
    
    business_metrics = calculator.calculate_business_metrics(empty_results, empty_df)
    assert isinstance(business_metrics, dict)


# Test FilterBubbleAnalyzer class
def test_filter_bubble_analyzer_initialization():
    """Test that FilterBubbleAnalyzer initializes correctly."""
    analyzer = FilterBubbleAnalyzer()
    assert analyzer is not None
    assert analyzer.metrics_calculator is not None


def test_analyze_bubble_formation():
    """Test bubble formation analysis."""
    analyzer = FilterBubbleAnalyzer()
    results = create_test_simulation_results()
    interactions_df = create_test_interactions_df()
    
    bubble_analysis = analyzer.analyze_bubble_formation(results, interactions_df)
    
    # Check that all expected sections are present
    expected_keys = [
        'bubble_metrics', 'diversity_timeline', 
        'user_segmentation', 'content_impact', 'bubble_phases'
    ]
    
    for key in expected_keys:
        assert key in bubble_analysis, f"Missing key: {key}"
    
    # Check diversity timeline structure
    timeline = bubble_analysis['diversity_timeline']
    assert 'initial_diversity' in timeline
    assert 'final_diversity' in timeline
    assert 'diversity_range' in timeline
    
    # Check user segmentation
    segmentation = bubble_analysis['user_segmentation']
    if 'bubble_susceptible_fraction' in segmentation:
        assert 0 <= segmentation['bubble_susceptible_fraction'] <= 1


def test_compare_bubble_prevention():
    """Test bubble prevention comparison."""
    analyzer = FilterBubbleAnalyzer()
    
    # Create contrasting results (one with bubbles, one without)
    traditional_results = create_test_simulation_results('Traditional')
    # Simulate stronger bubble formation
    traditional_results.content_diversity = [0.8 - i * 0.08 for i in range(10)]
    
    rl_results = create_test_simulation_results('RL')
    # Simulate better diversity preservation
    rl_results.content_diversity = [0.8 - i * 0.02 for i in range(10)]
    
    results_dict = {'Traditional': traditional_results, 'RL': rl_results}
    interactions_dict = {
        'Traditional': create_test_interactions_df(500, 25, 50, 10),
        'RL': create_test_interactions_df(500, 25, 80, 10)
    }
    
    comparison = analyzer.compare_bubble_prevention(results_dict, interactions_dict)
    
    # Check that all expected sections are present
    expected_keys = [
        'bubble_scores', 'best_bubble_prevention', 'worst_bubble_formation',
        'diversity_preservation', 'tradeoff_analysis'
    ]
    
    for key in expected_keys:
        assert key in comparison, f"Missing key: {key}"
    
    # Check that bubble scores are calculated
    assert 'Traditional' in comparison['bubble_scores']
    assert 'RL' in comparison['bubble_scores']
    
    # RL should have better bubble prevention (lower score)
    assert comparison['bubble_scores']['RL'] < comparison['bubble_scores']['Traditional']


def test_generate_bubble_report():
    """Test comprehensive bubble report generation."""
    analyzer = FilterBubbleAnalyzer()
    
    results_dict = {
        'Traditional': create_test_simulation_results('Traditional'),
        'RL': create_test_simulation_results('RL')
    }
    
    interactions_dict = {
        'Traditional': create_test_interactions_df(500, 25, 50, 10),
        'RL': create_test_interactions_df(500, 25, 80, 10)
    }
    
    report = analyzer.generate_bubble_report(results_dict, interactions_dict)
    
    # Check that all expected sections are present
    expected_keys = [
        'executive_summary', 'detailed_analysis', 
        'comparative_analysis', 'statistical_evidence', 'recommendations'
    ]
    
    for key in expected_keys:
        assert key in report, f"Missing key: {key}"
    
    # Check executive summary structure
    summary = report['executive_summary']
    assert 'worst_bubble_formation' in summary
    assert 'best_bubble_prevention' in summary
    assert 'diversity_statistics' in summary
    
    # Check that detailed analysis exists for both recommenders
    assert 'Traditional' in report['detailed_analysis']
    assert 'RL' in report['detailed_analysis']
    
    # Check that recommendations are provided
    recommendations = report['recommendations']
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0


def test_bubble_analyzer_helper_methods():
    """Test helper methods in FilterBubbleAnalyzer."""
    analyzer = FilterBubbleAnalyzer()
    
    # Test bubble score calculation
    good_results = create_test_simulation_results()
    good_results.content_diversity = [0.8, 0.8, 0.8, 0.8, 0.8]  # Stable diversity
    
    bad_results = create_test_simulation_results()
    bad_results.content_diversity = [0.8, 0.6, 0.4, 0.2, 0.1]  # Severe decline
    
    good_score = analyzer._calculate_bubble_score(good_results)
    bad_score = analyzer._calculate_bubble_score(bad_results)
    
    # Bad results should have higher bubble score
    assert bad_score > good_score
    
    # Test diversity preservation calculation
    good_preservation = analyzer._calculate_diversity_preservation(good_results)
    bad_preservation = analyzer._calculate_diversity_preservation(bad_results)
    
    # Good results should have better preservation
    assert good_preservation > bad_preservation


def test_statistical_significance():
    """Test statistical significance calculations."""
    calculator = MetricsCalculator()
    
    # Create significantly different results
    results1 = create_test_simulation_results('System1')
    results1.content_diversity = [0.8 - i * 0.01 for i in range(10)]  # Slight decline
    results1.engagement_rate = [0.6 + i * 0.01 for i in range(10)]  # Slight increase
    
    results2 = create_test_simulation_results('System2')
    results2.content_diversity = [0.8 - i * 0.05 for i in range(10)]  # Steep decline
    results2.engagement_rate = [0.5 + i * 0.005 for i in range(10)]  # Slower increase
    
    results_dict = {'System1': results1, 'System2': results2}
    interactions_dict = {
        'System1': create_test_interactions_df(500),
        'System2': create_test_interactions_df(500)
    }
    
    tests = calculator._perform_significance_tests(results_dict, interactions_dict)
    
    # Check that tests were performed
    assert 'diversity_tests' in tests
    assert 'engagement_tests' in tests
    
    # Check that comparison exists
    if 'System1_vs_System2' in tests['diversity_tests']:
        test_result = tests['diversity_tests']['System1_vs_System2']
        assert 'statistic' in test_result
        assert 'p_value' in test_result
        assert 'significant' in test_result
        assert isinstance(test_result['significant'], (bool, np.bool_))


def test_metrics_with_missing_columns():
    """Test metrics calculation with missing DataFrame columns."""
    calculator = MetricsCalculator()
    results = create_test_simulation_results()
    
    # Create DataFrame with missing columns
    minimal_df = pd.DataFrame({
        'timestep': [0, 1, 2],
        'user_id': [0, 1, 2],
        'item_id': [0, 1, 2],
        'engaged': [True, False, True]
    })
    
    # Should not crash with missing columns
    diversity_metrics = calculator.calculate_diversity_metrics(results, minimal_df)
    engagement_metrics = calculator.calculate_engagement_metrics(results, minimal_df)
    business_metrics = calculator.calculate_business_metrics(results, minimal_df)
    
    # Should return valid dictionaries
    assert isinstance(diversity_metrics, dict)
    assert isinstance(engagement_metrics, dict)
    assert isinstance(business_metrics, dict)


def test_value_ranges():
    """Test that calculated metrics are within expected ranges."""
    calculator = MetricsCalculator()
    results = create_test_simulation_results()
    interactions_df = create_test_interactions_df()
    
    # Calculate all metrics
    diversity_metrics = calculator.calculate_diversity_metrics(results, interactions_df)
    engagement_metrics = calculator.calculate_engagement_metrics(results, interactions_df)
    business_metrics = calculator.calculate_business_metrics(results, interactions_df)
    bubble_metrics = calculator.calculate_bubble_formation_metrics(results, interactions_df)
    
    # Check that probabilities/rates are between 0 and 1
    rate_metrics = [
        diversity_metrics['content_diversity_final'],
        diversity_metrics['diversity_gini_coefficient'],
        diversity_metrics['long_tail_exposure'],
        engagement_metrics['engagement_rate_final'],
        engagement_metrics['position_bias_impact'],
        engagement_metrics['repeat_engagement_rate'],
        business_metrics['catalog_coverage_final'],
        business_metrics['user_retention_rate'],
        business_metrics['niche_content_performance'],
        bubble_metrics['echo_chamber_strength']
    ]
    
    for metric in rate_metrics:
        assert 0 <= metric <= 1, f"Metric {metric} not in range [0, 1]"
    
    # Check that counts/rates are non-negative
    positive_metrics = [
        diversity_metrics['bubble_formation_rate'],
        engagement_metrics['user_engagement_variance'],
        business_metrics['catalog_utilization_efficiency'],
        business_metrics['revenue_per_user'],
        business_metrics['content_turnover_rate'],
        bubble_metrics['bubble_formation_velocity'],
        bubble_metrics['interest_narrowing_rate'],
        bubble_metrics['novelty_decay_rate']
    ]
    
    for metric in positive_metrics:
        assert metric >= 0, f"Metric {metric} should be non-negative"