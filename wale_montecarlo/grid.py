"""
Grid generation for Monte Carlo parameter sweep.

Creates the Cartesian product of all perturbation parameter values
to form the full grid of cells to evaluate.

Default grid: 6,048 cells (7 × 8 × 4 × 3 × 3 × 3)
With fixed delay=1: ~1,500 cells
"""

from typing import List, Dict, Optional, Tuple
from itertools import product

from .models import CellConfig, RunConfig, ShuffleMode, BootstrapMode


# Default parameter ranges (from documentation)
DEFAULT_P_SKIP = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
DEFAULT_SLIP = [0, 25, 50, 75, 100, 150, 200, 300]
DEFAULT_DELAY = [0, 1, 2, 3]
DEFAULT_SHUFFLE = [ShuffleMode.NONE, ShuffleMode.PERMUTE, ShuffleMode.BLOCK_PERMUTE]
DEFAULT_BOOTSTRAP = [BootstrapMode.NONE, BootstrapMode.TRADE_BOOTSTRAP, BootstrapMode.BLOCK_BOOTSTRAP]
DEFAULT_BLOCK_LEN = [5, 10, 20]


def generate_grid(config: RunConfig) -> List[CellConfig]:
    """
    Generate full grid of cell configurations.

    Creates Cartesian product of all parameter values.

    Args:
        config: RunConfig with parameter ranges

    Returns:
        List of CellConfig objects, one per grid cell
    """
    # Get parameter values from config
    p_skip_values = config.p_skip_values
    slip_values = config.slip_values
    delay_values = config.delay_values if config.fixed_delay is None else [config.fixed_delay]
    shuffle_modes = config.shuffle_modes
    bootstrap_modes = config.bootstrap_modes
    block_len_values = config.block_len_values

    cells = []

    # Generate Cartesian product
    for p_skip, slip, delay, shuffle, bootstrap, block_len in product(
        p_skip_values,
        slip_values,
        delay_values,
        shuffle_modes,
        bootstrap_modes,
        block_len_values
    ):
        cell = CellConfig(
            p_skip=p_skip,
            slip_dollars=slip,
            delay_bars_max=delay,
            shuffle_mode=shuffle,
            bootstrap_mode=bootstrap,
            block_len=block_len
        )
        cells.append(cell)

    return cells


def generate_grid_default(fixed_delay: Optional[int] = None) -> List[CellConfig]:
    """
    Generate grid with default parameter ranges.

    Args:
        fixed_delay: If set, fix delay to this value

    Returns:
        List of CellConfig objects
    """
    config = RunConfig(
        input_dir="",
        output_dir="",
        fixed_delay=fixed_delay,
        p_skip_values=DEFAULT_P_SKIP,
        slip_values=DEFAULT_SLIP,
        delay_values=DEFAULT_DELAY,
        shuffle_modes=DEFAULT_SHUFFLE,
        bootstrap_modes=DEFAULT_BOOTSTRAP,
        block_len_values=DEFAULT_BLOCK_LEN
    )
    return generate_grid(config)


def filter_grid(
    cells: List[CellConfig],
    filters: Optional[Dict] = None
) -> List[CellConfig]:
    """
    Filter grid cells based on criteria.

    Filters can include:
    - p_skip_max: Maximum p_skip value
    - slip_max: Maximum slippage
    - exclude_shuffle: List of shuffle modes to exclude
    - exclude_bootstrap: List of bootstrap modes to exclude
    - require_perturbation: If True, exclude baseline (all zeros) cell

    Args:
        cells: List of CellConfig objects
        filters: Dictionary of filter criteria

    Returns:
        Filtered list of cells
    """
    if filters is None:
        return cells

    result = []

    for cell in cells:
        include = True

        # p_skip filter
        if 'p_skip_max' in filters:
            if cell.p_skip > filters['p_skip_max']:
                include = False

        # Slippage filter
        if 'slip_max' in filters:
            if cell.slip_dollars > filters['slip_max']:
                include = False

        # Shuffle mode exclusions
        if 'exclude_shuffle' in filters:
            if cell.shuffle_mode.value in filters['exclude_shuffle']:
                include = False

        # Bootstrap mode exclusions
        if 'exclude_bootstrap' in filters:
            if cell.bootstrap_mode.value in filters['exclude_bootstrap']:
                include = False

        # Block length filter
        if 'block_len_values' in filters:
            if cell.block_len not in filters['block_len_values']:
                include = False

        # Require some perturbation (exclude baseline)
        if filters.get('require_perturbation', False):
            is_baseline = (
                cell.p_skip == 0 and
                cell.slip_dollars == 0 and
                cell.delay_bars_max == 0 and
                cell.shuffle_mode == ShuffleMode.NONE and
                cell.bootstrap_mode == BootstrapMode.NONE
            )
            if is_baseline:
                include = False

        if include:
            result.append(cell)

    return result


def cell_to_id(config: CellConfig) -> str:
    """
    Generate unique identifier for a cell.

    Args:
        config: CellConfig object

    Returns:
        String identifier
    """
    return config.to_cell_id()


def id_to_cell(cell_id: str) -> Optional[CellConfig]:
    """
    Parse cell ID back to CellConfig (if possible).

    Args:
        cell_id: String identifier

    Returns:
        CellConfig or None if parsing fails
    """
    try:
        # Parse format: skip0.05_slip100_delay1_shufpermute_bootblock_bootstrap_blk10
        parts = cell_id.split('_')

        p_skip = float(parts[0].replace('skip', ''))
        slip = float(parts[1].replace('slip', ''))
        delay = int(parts[2].replace('delay', ''))
        shuffle_str = parts[3].replace('shuf', '')
        bootstrap_str = parts[4].replace('boot', '')
        block_len = int(parts[5].replace('blk', ''))

        return CellConfig(
            p_skip=p_skip,
            slip_dollars=slip,
            delay_bars_max=delay,
            shuffle_mode=ShuffleMode(shuffle_str),
            bootstrap_mode=BootstrapMode(bootstrap_str),
            block_len=block_len
        )
    except Exception:
        return None


def get_grid_summary(cells: List[CellConfig]) -> Dict:
    """
    Get summary statistics for a grid.

    Args:
        cells: List of CellConfig objects

    Returns:
        Dictionary with grid statistics
    """
    if not cells:
        return {'total_cells': 0}

    p_skips = set(c.p_skip for c in cells)
    slips = set(c.slip_dollars for c in cells)
    delays = set(c.delay_bars_max for c in cells)
    shuffles = set(c.shuffle_mode for c in cells)
    bootstraps = set(c.bootstrap_mode for c in cells)
    block_lens = set(c.block_len for c in cells)

    return {
        'total_cells': len(cells),
        'p_skip_values': sorted(p_skips),
        'slip_values': sorted(slips),
        'delay_values': sorted(delays),
        'shuffle_modes': [m.value for m in shuffles],
        'bootstrap_modes': [m.value for m in bootstraps],
        'block_len_values': sorted(block_lens),
        'dimensions': {
            'p_skip': len(p_skips),
            'slip': len(slips),
            'delay': len(delays),
            'shuffle': len(shuffles),
            'bootstrap': len(bootstraps),
            'block_len': len(block_lens)
        }
    }


def estimate_runtime(
    n_cells: int,
    n_perms: int,
    trades_per_perm: int = 500,
    perms_per_second: float = 10000.0
) -> Dict:
    """
    Estimate runtime for a grid search.

    Args:
        n_cells: Number of grid cells
        n_perms: Permutations per cell
        trades_per_perm: Average trades per permutation
        perms_per_second: Estimated permutations per second per core

    Returns:
        Runtime estimates
    """
    total_perms = n_cells * n_perms

    # Single-threaded estimate
    single_core_seconds = total_perms / perms_per_second
    single_core_hours = single_core_seconds / 3600

    return {
        'total_permutations': total_perms,
        'single_core_hours': single_core_hours,
        '8_core_hours': single_core_hours / 8,
        '16_core_hours': single_core_hours / 16,
        'cells': n_cells,
        'perms_per_cell': n_perms
    }


def split_grid_for_parallel(
    cells: List[CellConfig],
    n_chunks: int
) -> List[List[CellConfig]]:
    """
    Split grid into chunks for parallel processing.

    Args:
        cells: List of cells
        n_chunks: Number of chunks

    Returns:
        List of cell lists (chunks)
    """
    if n_chunks <= 0:
        n_chunks = 1

    chunk_size = max(1, len(cells) // n_chunks)
    chunks = []

    for i in range(0, len(cells), chunk_size):
        chunks.append(cells[i:i + chunk_size])

    # Merge last chunk if too small
    if len(chunks) > n_chunks and len(chunks[-1]) < chunk_size // 2:
        chunks[-2].extend(chunks[-1])
        chunks = chunks[:-1]

    return chunks
