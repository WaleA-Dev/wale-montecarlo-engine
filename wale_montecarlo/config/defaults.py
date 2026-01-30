"""
Default configuration for the Monte Carlo engine.

Grid presets, scoring parameters, and overfit thresholds.
"""

# Exploration grid: 128 cells for quick runs (2-4 hours)
EXPLORATION_GRID = {
    'p_skip': [0.0, 0.02, 0.05, 0.10],
    'slip': [0, 50, 100, 200],
    'delay': [0, 1],
    'shuffle': ['none', 'permute'],
    'bootstrap': ['none', 'trade_bootstrap'],
    'block_len': [10],
}

# Focus grid: use with parameter ranges after exploration
FOCUS_GRID = {
    'p_skip': [0.02, 0.03, 0.05, 0.08],
    'slip': [50, 75, 100, 150],
    'delay': [1],
    'shuffle': ['none', 'permute'],
    'bootstrap': ['none', 'trade_bootstrap'],
    'block_len': [10],
}

# Full grid: 6,048 cells for publication quality (24-48 hours)
FULL_GRID = {
    'p_skip': [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
    'slip': [0, 25, 50, 75, 100, 150, 200, 300],
    'delay': [0, 1, 2, 3],
    'shuffle': ['none', 'permute', 'block_permute'],
    'bootstrap': ['none', 'trade_bootstrap', 'block_bootstrap'],
    'block_len': [5, 10, 20],
}

# Default configuration
DEFAULT_CONFIG = {
    # Performance
    'n_workers': 'auto',
    'batch_size': 1000,
    'write_interval': 10000,
    
    # Permutations by mode
    'perms_per_mode': {
        'explore': 20000,
        'focus': 100000,
        'full': 200000,
    },
    
    # P-value correction
    'p_value_correction': 'bh',  # 'bonferroni', 'bh', or 'by'
    'confidence_level': 0.95,
    
    # Robust score v3 parameters
    'robust_score_version': 'v3',
    'maxdd_penalty_start': 0.20,  # penalty starts at 20% drawdown
    'maxdd_penalty_end': 0.60,    # score = 0 at 60% drawdown
    
    # Overfitting classification thresholds
    'overfit_thresholds': {
        'robust': 1.5,    # PF_P50 > 1.5 at moderate stress
        'fragile': 1.0,   # PF_P50 between 1.0-1.5 at moderate stress
    },
    
    # Moderate stress parameters for overfit detection
    'moderate_stress_params': {
        'p_skip': 0.05,
        'slip': 100,
        'delay': 1,
    },
}


def get_grid_config(mode: str) -> dict:
    """
    Get grid configuration for a mode.
    
    Args:
        mode: 'explore', 'focus', or 'full'
    
    Returns:
        Grid configuration dict
    """
    if mode == 'explore':
        return EXPLORATION_GRID.copy()
    elif mode == 'focus':
        return FOCUS_GRID.copy()
    elif mode == 'full':
        return FULL_GRID.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'explore', 'focus', or 'full'.")


def get_perms_for_mode(mode: str) -> int:
    """Get default permutation count for a mode."""
    return DEFAULT_CONFIG['perms_per_mode'].get(mode, 50000)


def count_grid_cells(grid_config: dict) -> int:
    """Count total cells in a grid configuration."""
    total = 1
    for values in grid_config.values():
        total *= len(values)
    return total
