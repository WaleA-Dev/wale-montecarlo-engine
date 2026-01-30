"""
Perturbation interaction analysis.

Analyzes how perturbations combine and interact to affect performance.
Uses ANOVA-style decomposition to identify significant interactions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def analyze_interactions(
    grid_data: List[Dict],
    target_metric: str = 'pf_p50'
) -> Dict:
    """
    Analyze perturbation interactions using linear regression.
    
    Fits a model with main effects and 2-way interactions to understand
    how perturbations combine.
    
    Args:
        grid_data: List of cell dicts with perturbation parameters and metrics
        target_metric: Which metric to analyze (default: 'pf_p50')
    
    Returns:
        Dictionary with:
        - main_effects: Dict of parameter -> coefficient
        - interactions: Dict of 'param1:param2' -> coefficient and p-value
        - significant_interactions: List of significant interactions (p < 0.05)
        - model_r2: R-squared of the fitted model
    
    Raises:
        ImportError: If pandas or statsmodels not available
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for interaction analysis")
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for interaction analysis")
    
    if len(grid_data) < 10:
        return {
            'error': 'Insufficient data for interaction analysis (need >= 10 cells)',
            'main_effects': {},
            'interactions': {},
            'significant_interactions': [],
            'model_r2': 0.0
        }
    
    # Convert to DataFrame
    df = pd.DataFrame(grid_data)
    
    # Ensure target metric exists
    if target_metric not in df.columns:
        return {
            'error': f'Target metric {target_metric} not found in data',
            'main_effects': {},
            'interactions': {},
            'significant_interactions': [],
            'model_r2': 0.0
        }
    
    # Create dummy variables for categorical factors
    if 'shuffle_mode' in df.columns:
        df['shuffle_permute'] = (df['shuffle_mode'].astype(str).str.lower() == 'permute').astype(int)
    else:
        df['shuffle_permute'] = 0
        
    if 'bootstrap_mode' in df.columns:
        df['bootstrap_trade'] = (df['bootstrap_mode'].astype(str).str.lower().str.contains('trade')).astype(int)
    else:
        df['bootstrap_trade'] = 0
    
    # Standardize column names
    if 'slip_dollars' in df.columns:
        df['slip'] = df['slip_dollars']
    if 'delay_bars_max' in df.columns:
        df['delay'] = df['delay_bars_max']
    
    # Build formula with available columns
    numeric_vars = []
    for var in ['p_skip', 'slip', 'delay']:
        if var in df.columns and df[var].notna().any():
            numeric_vars.append(var)
    
    categorical_vars = []
    for var in ['shuffle_permute', 'bootstrap_trade']:
        if var in df.columns and df[var].sum() > 0:
            categorical_vars.append(var)
    
    all_vars = numeric_vars + categorical_vars
    
    if len(all_vars) < 2:
        return {
            'error': 'Need at least 2 varying parameters for interaction analysis',
            'main_effects': {},
            'interactions': {},
            'significant_interactions': [],
            'model_r2': 0.0
        }
    
    # Build formula: main effects + 2-way interactions
    main_terms = ' + '.join(all_vars)
    
    # Generate 2-way interaction terms
    interaction_terms = []
    for i, var1 in enumerate(all_vars):
        for var2 in all_vars[i+1:]:
            interaction_terms.append(f'{var1}:{var2}')
    
    formula = f'{target_metric} ~ {main_terms}'
    if interaction_terms:
        formula += ' + ' + ' + '.join(interaction_terms)
    
    try:
        model = ols(formula, data=df).fit()
    except Exception as e:
        return {
            'error': f'Model fitting failed: {str(e)}',
            'main_effects': {},
            'interactions': {},
            'significant_interactions': [],
            'model_r2': 0.0
        }
    
    # Extract results
    main_effects = {}
    interactions = {}
    significant_interactions = []
    
    for name in model.params.index:
        if name == 'Intercept':
            continue
        
        coef = float(model.params[name])
        pval = float(model.pvalues[name])
        
        if ':' in name:
            # Interaction term
            interactions[name] = {
                'coefficient': coef,
                'p_value': pval,
                'significant': pval < 0.05
            }
            if pval < 0.05:
                significant_interactions.append(name)
        else:
            # Main effect
            main_effects[name] = {
                'coefficient': coef,
                'p_value': pval,
                'significant': pval < 0.05
            }
    
    return {
        'main_effects': main_effects,
        'interactions': interactions,
        'significant_interactions': significant_interactions,
        'model_r2': float(model.rsquared),
        'n_observations': len(df),
        'formula': formula
    }


def interaction_adjusted_prediction(
    cell_params: Dict,
    interaction_model: Dict,
    baseline_pf: float = 1.0
) -> float:
    """
    Predict performance adjusted for interaction effects.
    
    Args:
        cell_params: Parameter values for the cell
        interaction_model: Result from analyze_interactions()
        baseline_pf: Baseline profit factor
    
    Returns:
        Predicted profit factor accounting for interactions
    """
    if 'error' in interaction_model:
        return baseline_pf
    
    prediction = baseline_pf
    
    # Add main effects
    for param, effect in interaction_model.get('main_effects', {}).items():
        value = cell_params.get(param, 0)
        prediction += value * effect['coefficient']
    
    # Add interaction effects
    for interaction, effect in interaction_model.get('interactions', {}).items():
        parts = interaction.split(':')
        if len(parts) == 2:
            val1 = cell_params.get(parts[0], 0)
            val2 = cell_params.get(parts[1], 0)
            prediction += val1 * val2 * effect['coefficient']
    
    return prediction


def summarize_interactions(interaction_results: Dict) -> str:
    """
    Generate human-readable summary of interaction analysis.
    
    Args:
        interaction_results: Result from analyze_interactions()
    
    Returns:
        Markdown-formatted summary string
    """
    if 'error' in interaction_results:
        return f"**Interaction Analysis Error:** {interaction_results['error']}"
    
    lines = [
        "## Perturbation Interaction Analysis",
        "",
        f"Model R²: {interaction_results['model_r2']:.3f}",
        f"Observations: {interaction_results['n_observations']}",
        "",
        "### Main Effects",
        ""
    ]
    
    for param, effect in interaction_results.get('main_effects', {}).items():
        sig = "**" if effect['significant'] else ""
        lines.append(f"- {sig}{param}{sig}: {effect['coefficient']:.4f} (p={effect['p_value']:.4f})")
    
    lines.extend(["", "### Interaction Effects", ""])
    
    interactions = interaction_results.get('interactions', {})
    if interactions:
        for name, effect in interactions.items():
            sig = "**" if effect['significant'] else ""
            lines.append(f"- {sig}{name}{sig}: {effect['coefficient']:.4f} (p={effect['p_value']:.4f})")
    else:
        lines.append("No significant interactions found.")
    
    significant = interaction_results.get('significant_interactions', [])
    if significant:
        lines.extend([
            "",
            "### Key Finding",
            "",
            f"Significant interactions detected: **{', '.join(significant)}**",
            "",
            "This means perturbation effects are not fully additive - the combined impact ",
            "may be greater or less than the sum of individual effects."
        ])
    else:
        lines.extend([
            "",
            "### Key Finding",
            "",
            "No significant interactions detected (p < 0.05).",
            "Perturbation effects can be treated as approximately additive."
        ])
    
    return '\n'.join(lines)
