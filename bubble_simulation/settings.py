"""
Central configuration file for all simulation parameters.
All numerical constants and configuration options should be defined here.
"""

#------------------------------------------------------------------------------
# GENERAL SIMULATION SETTINGS
#------------------------------------------------------------------------------
DEFAULT_SEED = 42  # Default random seed for reproducibility

#------------------------------------------------------------------------------
# CONTENT MODULE SETTINGS
#------------------------------------------------------------------------------
# ContentUniverse default settings
CONTENT_NUM_ITEMS = 5000  # Default number of items in the content universe
CONTENT_NUM_CATEGORIES = 20  # Default number of content categories
CONTENT_NUM_FEATURES = 50  # Feature vector dimensionality
CONTENT_POPULARITY_POWER_LAW = 1.5  # Exponent for power law distribution
CONTENT_GROWTH_RATE = 0.01  # Rate of new content addition per timestep

# Item similarity settings
CONTENT_FEATURE_NOISE = 0.5  # Noise level when generating item features
CONTENT_MULTI_CATEGORY_PROBABILITY = 0.3  # Probability of assigning multiple categories

#------------------------------------------------------------------------------
# USER MODULE SETTINGS
#------------------------------------------------------------------------------
# UserUniverse default settings
USER_NUM_USERS = 1000  # Default number of users in the simulation
USER_NUM_FEATURES = 50  # Should match content feature dimension

# User trait distribution parameters
USER_EXPLORATION_FACTOR_MEAN = 0.2  # Mean value for exploration tendency
USER_EXPLORATION_FACTOR_STD = 0.1  # Standard deviation for exploration tendency
USER_POSITION_BIAS_FACTOR_MEAN = 0.8  # Mean value for position bias susceptibility
USER_POSITION_BIAS_FACTOR_STD = 0.1  # Standard deviation for position bias
USER_DIVERSITY_PREFERENCE_MEAN = 0.5  # Mean value for diversity preference
USER_DIVERSITY_PREFERENCE_STD = 0.15  # Standard deviation for diversity preference
USER_ATTENTION_SPAN = 10  # Default user attention span (items considered)

# User preference updating parameters
USER_PREFERENCE_ADAPTATION_RATE = 0.05  # How quickly preferences adapt after interaction
USER_PREFERENCE_FEATURE_WEIGHT = 0.7  # Weight given to feature similarity in preference
USER_PREFERENCE_CATEGORY_WEIGHT = 0.3  # Weight given to category interest in preference

# EngagementModel default parameters
ENGAGEMENT_BASE_RATE = 0.3  # Base engagement probability
ENGAGEMENT_POSITION_DECAY = 0.85  # Position bias decay factor
ENGAGEMENT_NOVELTY_FACTOR = 0.2  # Impact of content novelty on engagement
ENGAGEMENT_DIVERSITY_FACTOR = 0.1  # Impact of category diversity on engagement
ENGAGEMENT_TIME_PENALTY = 0.05  # Penalty for recently seen content

# Interaction history parameters
RECENT_HISTORY_SIZE = 10  # Number of recent interactions to consider for novelty
HISTORY_MAX_SIZE = 100  # Maximum size of user interaction history

#------------------------------------------------------------------------------
# RECOMMENDATION MODULE SETTINGS
#------------------------------------------------------------------------------
# General recommendation settings
RECO_TOP_K = 10  # Default number of recommendations to generate
RECO_TRAINING_BATCH_SIZE = 128  # Batch size for training recommendation models
RECO_LEARNING_RATE = 0.01  # Learning rate for optimization
RECO_REGULARIZATION = 0.001  # Regularization factor to prevent overfitting
RECO_EMBEDDING_SIZE = 50  # Size of latent embeddings

# Collaborative filtering settings
CF_NUM_FACTORS = 50  # Dimensionality of latent factors
CF_INIT_SCALE = 0.1  # Scale for random initialization
CF_RETRAIN_FREQUENCY = 10  # How often to retrain the model (in timesteps)

#------------------------------------------------------------------------------
# RL RECOMMENDATION SETTINGS
#------------------------------------------------------------------------------
# Basic RL parameters
RL_EXPLORATION_EPSILON = 0.1  # Epsilon for epsilon-greedy exploration
RL_DISCOUNT_FACTOR = 0.9  # Discount factor for future rewards

# Reward function weights
RL_REWARD_ENGAGEMENT_WEIGHT = 0.7  # Weight for engagement in reward function
RL_REWARD_DIVERSITY_WEIGHT = 0.3  # Weight for diversity in reward function
RL_REWARD_REVENUE_WEIGHT = 0.1  # Weight for revenue in reward function
RL_REWARD_RETENTION_WEIGHT = 0.1  # Weight for retention in reward function
RL_STRATEGIC_BONUS = 0.2  # Bonus for strategic items

# Learning parameters
RL_LEARNING_RATE = 0.1  # Learning rate for Q-value updates
RL_FEATURE_BINS = 10  # Number of bins for state discretization 
RL_MEMORY_SIZE = 10000  # Size of replay memory
RL_BATCH_SIZE = 32  # Batch size for training
RL_EPSILON_DECAY = 0.995  # Rate at which epsilon decays
RL_MIN_EPSILON = 0.01  # Minimum exploration rate

# Advanced exploration settings
RL_ENABLE_STAGED_EXPLORATION = True  # Whether to use staged exploration
RL_ENABLE_BUSINESS_AWARE_EXPLORATION = True  # Whether to use business-aware exploration
RL_NEW_USER_THRESHOLD = 20  # Interaction threshold for new users
RL_ESTABLISHED_USER_THRESHOLD = 100  # Interaction threshold for established users
RL_NEW_USER_EPSILON_BOOST = 2.0  # Multiplier for new users' exploration rate
RL_ESTABLISHED_USER_EPSILON_REDUCTION = 0.5  # Multiplier for established users' exploration rate
RL_HISTORY_WINDOW_SIZE = 10  # Number of recent interactions to consider for diversity

RL_CATEGORY_INTEREST_SIZE = 5  # Fixed size for category interests in state
RL_STATE_SAMPLE_SIZE = 100  # Number of users to sample for state scaler
RL_HIGH_RETENTION_CATEGORIES = {0, 2, 5}  # Categories with higher retention impact
RL_BASE_RETENTION_SCORE = 0.2  # Base retention score
RL_RETENTION_CATEGORY_BONUS = 0.2  # Bonus per retention category
RL_REVENUE_POPULARITY_FACTOR = 0.5  # Factor for default revenue calculation

#------------------------------------------------------------------------------
# SIMULATION SETTINGS
#------------------------------------------------------------------------------
SIM_NUM_TIMESTEPS = 100  # Default number of simulation timesteps
SIM_INITIAL_INTERACTIONS_PER_USER = 20  # Initial interactions to bootstrap
SIM_RECOMMENDATIONS_PER_STEP = 10  # Recommendations per user per timestep