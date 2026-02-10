"""
Analytics module - performance tracking and reporting
"""

import pandas as pd
from typing import Dict, List
from datetime import datetime


class PerformanceTracker:
    """Tracks trading performance metrics"""

    def __init__(self):
        self.trades: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}

    def add_trade(self, trade: Dict):
        """Record a completed trade"""
        self.trades.append(trade)

    def get_summary(self) -> Dict:
        """Get performance summary"""
        if not self.trades:
            return {}

        df = pd.DataFrame(self.trades)

        winning_trades = df[df["pnl"] > 0]
        losing_trades = df[df["pnl"] < 0]

        return {
            "total_trades": len(df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(df) if len(df) > 0 else 0,
            "avg_win": winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0,
            "avg_loss": losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0,
            "total_pnl": df["pnl"].sum(),
            "max_drawdown": self._calculate_max_drawdown(df),
        }

    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        cumsum = df["pnl"].cumsum()
        running_max = cumsum.expanding().max()
        drawdown = (cumsum - running_max) / running_max
        return drawdown.min() if len(drawdown) > 0 else 0
