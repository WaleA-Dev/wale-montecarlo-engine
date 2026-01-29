"""
Parallel runner for Monte Carlo grid search.

Orchestrates multiple worker processes to run the full grid:
- Manages process pool
- Tracks overall progress
- Handles interrupts gracefully
- Generates heartbeat for monitoring
"""

import os
import sys
import time
import json
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

from .models import (
    Trade, CellConfig, RunConfig, OHLCData,
    CellProgress, RunProgress, PermutationResult
)
from .grid import generate_grid, filter_grid, get_grid_summary
from .worker import run_cell, get_cell_status
from .io import (
    load_trade_list, load_equity_curve, load_ohlc_data,
    save_run_manifest, save_progress, save_heartbeat, save_done_sentinel,
    ensure_output_structure, get_cell_dir, atomic_write_json
)


logger = logging.getLogger(__name__)


# Global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown requested, finishing current cells...")


class MonteCarloRunner:
    """
    Main runner class for Monte Carlo grid search.

    Manages the full lifecycle:
    1. Load input data
    2. Generate grid
    3. Run cells in parallel
    4. Track progress
    5. Generate final report
    """

    def __init__(self, config: RunConfig):
        """
        Initialize runner with configuration.

        Args:
            config: RunConfig with all settings
        """
        self.config = config
        self.trades: List[Trade] = []
        self.ohlc_data: Optional[OHLCData] = None
        self.cells: List[CellConfig] = []
        self.cell_progress: Dict[str, CellProgress] = {}
        self.run_progress: Optional[RunProgress] = None
        self.paths: Dict[str, str] = {}

        # Set up signal handlers
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    def setup(self) -> bool:
        """
        Set up the run: load data, generate grid, create output dirs.

        Returns:
            True if setup successful
        """
        try:
            # Load trade list
            trade_list_path = os.path.join(self.config.input_dir, "trade_list.csv")
            if not os.path.exists(trade_list_path):
                logger.error(f"Trade list not found: {trade_list_path}")
                return False

            self.trades = load_trade_list(trade_list_path)
            logger.info(f"Loaded {len(self.trades)} trades")

            # Load OHLC data if available
            ohlc_paths = [
                os.path.join(self.config.input_dir, "jan_2_data_to_now.csv"),
                os.path.join(self.config.input_dir, "ohlc.csv"),
                os.path.join(self.config.input_dir, "price_data.csv"),
            ]
            for ohlc_path in ohlc_paths:
                if os.path.exists(ohlc_path):
                    self.ohlc_data = load_ohlc_data(ohlc_path)
                    logger.info(f"Loaded OHLC data: {len(self.ohlc_data.bars)} bars")
                    break

            # Generate grid
            self.cells = generate_grid(self.config)

            # Apply filters if specified
            if self.config.grid_filters:
                self.cells = filter_grid(self.cells, self.config.grid_filters)

            logger.info(f"Generated grid: {len(self.cells)} cells")

            # Create output structure
            self.paths = ensure_output_structure(self.config.output_dir)

            # Save manifest
            manifest_path = os.path.join(self.paths["aggregated"], "run_manifest.json")
            save_run_manifest(manifest_path, self.config, self.cells)

            # Initialize progress tracking
            self._init_progress()

            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False

    def _init_progress(self) -> None:
        """Initialize progress tracking for all cells."""
        for cell in self.cells:
            cell_id = cell.to_cell_id()
            cell_dir = get_cell_dir(self.config.output_dir, cell_id)

            # Check existing status
            status = get_cell_status(cell_dir, self.config.n_per_cell)

            self.cell_progress[cell_id] = CellProgress(
                cell_id=cell_id,
                status="completed" if status["is_complete"] else "pending",
                n_completed=status["completed"],
                n_target=self.config.n_per_cell
            )

        # Overall progress
        completed = sum(1 for cp in self.cell_progress.values() if cp.status == "completed")

        self.run_progress = RunProgress(
            run_name=os.path.basename(self.config.output_dir),
            start_time=datetime.now(),
            cells_total=len(self.cells),
            cells_completed=completed,
            perms_total=len(self.cells) * self.config.n_per_cell,
            perms_completed=sum(cp.n_completed for cp in self.cell_progress.values())
        )

    def run(self, resume: bool = True) -> bool:
        """
        Run the Monte Carlo grid search.

        Args:
            resume: If True, skip completed cells

        Returns:
            True if completed successfully
        """
        global _shutdown_requested

        if not self.trades or not self.cells:
            logger.error("Must call setup() before run()")
            return False

        # Get cells that need processing
        cells_to_run = [
            cell for cell in self.cells
            if self.cell_progress[cell.to_cell_id()].status != "completed"
        ]

        logger.info(f"Cells to process: {len(cells_to_run)}/{len(self.cells)}")

        if not cells_to_run:
            logger.info("All cells already complete")
            self._finalize()
            return True

        # Start heartbeat thread
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        heartbeat_thread.start()

        # Run cells in parallel
        n_jobs = min(self.config.n_jobs, len(cells_to_run))
        logger.info(f"Starting {n_jobs} parallel workers")

        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all cells
                futures = {}
                for cell in cells_to_run:
                    if _shutdown_requested:
                        break

                    future = executor.submit(
                        _run_cell_wrapper,
                        cell,
                        self.trades,
                        self.config.n_per_cell,
                        self.config.output_dir,
                        self.ohlc_data,
                        resume
                    )
                    futures[future] = cell

                # Process results as they complete
                for future in as_completed(futures):
                    if _shutdown_requested:
                        logger.info("Cancelling remaining tasks...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    cell = futures[future]
                    cell_id = cell.to_cell_id()

                    try:
                        success, n_completed = future.result(timeout=3600)

                        if success:
                            self.cell_progress[cell_id].status = "completed"
                            self.cell_progress[cell_id].n_completed = n_completed
                            self.run_progress.cells_completed += 1
                            logger.info(f"Completed: {cell_id}")
                        else:
                            self.cell_progress[cell_id].status = "failed"
                            self.run_progress.cells_failed += 1
                            logger.warning(f"Failed: {cell_id}")

                    except Exception as e:
                        self.cell_progress[cell_id].status = "failed"
                        self.cell_progress[cell_id].error_message = str(e)
                        self.run_progress.cells_failed += 1
                        logger.error(f"Error in {cell_id}: {e}")

                    # Update progress
                    self._update_progress()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        # Finalize
        self._finalize()

        return self.run_progress.cells_failed == 0

    def _heartbeat_loop(self) -> None:
        """Background thread for heartbeat updates."""
        heartbeat_path = os.path.join(self.paths["aggregated"], "heartbeat.json")

        while not _shutdown_requested:
            try:
                self.run_progress.last_heartbeat = datetime.now()
                save_heartbeat(heartbeat_path, self.run_progress)
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")

            time.sleep(30)

    def _update_progress(self) -> None:
        """Update progress files."""
        try:
            # Update perms completed
            self.run_progress.perms_completed = sum(
                cp.n_completed for cp in self.cell_progress.values()
            )

            # Save progress CSV
            progress_path = os.path.join(self.paths["aggregated"], "progress.csv")
            save_progress(progress_path, self.run_progress, list(self.cell_progress.values()))

        except Exception as e:
            logger.debug(f"Progress update error: {e}")

    def _finalize(self) -> None:
        """Finalize the run."""
        # Final progress update
        self._update_progress()

        # Check if complete
        all_complete = all(
            cp.status == "completed" for cp in self.cell_progress.values()
        )

        if all_complete:
            # Write done sentinel
            done_path = os.path.join(self.paths["aggregated"], "DONE.txt")
            save_done_sentinel(done_path)
            logger.info("Run complete!")
        else:
            n_failed = sum(1 for cp in self.cell_progress.values() if cp.status == "failed")
            n_pending = sum(1 for cp in self.cell_progress.values() if cp.status == "pending")
            logger.info(f"Run incomplete: {n_failed} failed, {n_pending} pending")

    def get_status(self) -> Dict:
        """Get current run status."""
        return {
            "run_name": self.run_progress.run_name if self.run_progress else None,
            "cells_total": len(self.cells),
            "cells_completed": sum(1 for cp in self.cell_progress.values() if cp.status == "completed"),
            "cells_failed": sum(1 for cp in self.cell_progress.values() if cp.status == "failed"),
            "cells_pending": sum(1 for cp in self.cell_progress.values() if cp.status == "pending"),
            "perms_completed": sum(cp.n_completed for cp in self.cell_progress.values()),
            "perms_total": len(self.cells) * self.config.n_per_cell if self.cells else 0
        }


def _run_cell_wrapper(
    cell: CellConfig,
    trades: List[Trade],
    n_perms: int,
    output_dir: str,
    ohlc_data: Optional[OHLCData],
    resume: bool
) -> Tuple[bool, int]:
    """
    Wrapper for running a cell in a subprocess.

    Returns:
        Tuple of (success, n_completed)
    """
    try:
        results, summary = run_cell(
            cell,
            trades,
            n_perms,
            output_dir,
            ohlc_data,
            resume=resume
        )
        return True, len(results)
    except Exception as e:
        logger.error(f"Cell {cell.to_cell_id()} failed: {e}")
        return False, 0


def run_surface(config: RunConfig, resume: bool = True) -> Dict:
    """
    Convenience function to run full surface analysis.

    Args:
        config: RunConfig with all settings
        resume: Resume from existing progress

    Returns:
        Status dictionary
    """
    runner = MonteCarloRunner(config)

    if not runner.setup():
        return {"success": False, "error": "Setup failed"}

    success = runner.run(resume=resume)

    return {
        "success": success,
        "status": runner.get_status()
    }


def get_run_status(output_dir: str) -> Dict:
    """
    Get status of an existing run.

    Args:
        output_dir: Path to run output directory

    Returns:
        Status dictionary
    """
    aggregated = os.path.join(output_dir, "aggregated")

    status = {
        "exists": os.path.exists(output_dir),
        "is_complete": os.path.exists(os.path.join(aggregated, "DONE.txt")),
        "has_manifest": os.path.exists(os.path.join(aggregated, "run_manifest.json"))
    }

    # Load manifest for details
    manifest_path = os.path.join(aggregated, "run_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
            status["n_cells"] = manifest.get("grid", {}).get("total_cells", 0)
            status["n_per_cell"] = manifest.get("config", {}).get("n_per_cell", 0)

    # Load progress
    progress_path = os.path.join(aggregated, "progress.csv")
    if os.path.exists(progress_path):
        import csv
        with open(progress_path, newline='') as f:
            reader = csv.DictReader(f)
            cells = list(reader)
            status["cells_completed"] = sum(1 for c in cells if c.get("status") == "completed")
            status["cells_total"] = len(cells)

    # Load heartbeat
    heartbeat_path = os.path.join(aggregated, "heartbeat.json")
    if os.path.exists(heartbeat_path):
        with open(heartbeat_path) as f:
            heartbeat = json.load(f)
            status["last_heartbeat"] = heartbeat.get("timestamp")
            status["pct_complete"] = heartbeat.get("pct_complete", 0)

    return status
