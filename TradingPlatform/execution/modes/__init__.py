"""
Execution Modes Package

Sub-packages for different trading modes:
- backtest: Historical testing
- simulation: Paper trading
- real: Production (Phase 2+)
"""

from .backtest_mode import BacktestMode
from .simulation_mode import SimulationMode
from .real_mode import RealMode

__all__ = ["BacktestMode", "SimulationMode", "RealMode"]
