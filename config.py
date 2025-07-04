import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOConfig:
    DISCOUNT_FACTOR = 0.998
    GAE_LAMBDA = 0.95
    ENTROPY_COEFFICIENT = 0.01
    PPO_CLIPPING_EPSILON = 0.2
    ABC_CLIPPING_EPSILON = 0.2
    VALUE_LOSS_WEIGHT = 1.0
    ABC_LOSS_WEIGHT = 0.5
    LEARNING_RATE = 3e-4
    SAMPLE_REUSE = 3

class ActionConfig:
    ACTION_DIMS = 6
    ACTION_BINS = 11

class SelfPlayConfig:
    ALICE_GOAL_SETTING_STEPS = 100
    BOB_MAX_GOAL_SOLVING_STEPS = 200
    NUM_GOALS_PER_EPISODE = 5
    PAST_POLICY_PROB = 0.2  # prob of playing against a past version of the opponent

class HardwareConfig:
    NUM_WORKERS = 12
    BLOCK_BATCH_SIZE = 4096

class RewardConfig:
    # Bob's rewards
    OBJECT_PLACED_REWARD = 1.0
    OBJECT_MOVED_AWAY_PENALTY = -1.0
    GOAL_SUCCESS_REWARD = 5.0

    # Alice's rewards
    BOB_FAILURE_REWARD = 5.0
    VALID_GOAL_REWARD = 1.0
    OUT_OF_ZONE_PENALTY = -3.0

class GoalValidationConfig:
    """Thresholds for determining goal success."""
    SUCCESS_THRESHOLD_POS = 0.04
    SUCCESS_THRESHOLD_ROT = 0.2
