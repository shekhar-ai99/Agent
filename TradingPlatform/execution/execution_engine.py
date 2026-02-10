"""
Execution Engine

Coordinates strategy signals, risk management, order placement, and position management.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import pandas as pd

from core import BaseStrategy, Signal, StrategyContext, BaseMarket, BaseBroker, RiskManager, StrategySelector
from brokers.paper_broker import PaperBroker

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Orchestrates the complete trading cycle:
    1. Receive market context (price, indicators, regime, etc.)
    2. Select best strategies
    3. Generate signals
    4. Validate risk
    5. Place orders
    6. Manage positions
    7. Log results
    """

    def __init__(
        self,
        market: BaseMarket,
        broker: BaseBroker,
        risk_manager: RiskManager,
        strategy_selector: StrategySelector,
        available_strategies: Dict[str, BaseStrategy],
    ):
        """
        Args:
            market: Market instance (IndianMarket, CryptoMarket, etc.)
            broker: Broker instance (PaperBroker, etc.)
            risk_manager: RiskManager instance
            strategy_selector: StrategySelector instance
            available_strategies: Dict of strategy_name -> BaseStrategy instance
        """
        self.market = market
        self.broker = broker
        self.risk_manager = risk_manager
        self.strategy_selector = strategy_selector
        self.available_strategies = available_strategies
        self.logger = logging.getLogger("ExecutionEngine")
        
        # State tracking
        self.last_signals: Dict[str, Signal] = {}
        self.execution_log: List[Dict] = []

    def execute_cycle(self, context: StrategyContext) -> Optional[Dict]:
        """
        Execute a single trading cycle.
        
        Returns:
            Execution summary dict or None if no trade
        """
        timestamp = context.timestamp
        symbol = context.symbol
        
        # Log cycle start
        self.logger.info(
            f"Cycle: {timestamp} | {symbol} | {context.regime.value} | "
            f"Volatility: {context.volatility.value}"
        )
        
        # === STEP 1: Check market hours ===
        if not self.market.is_trading_hours(timestamp):
            self.logger.info("Market is closed")
            return None
        
        # === STEP 2: Select strategies for current context ===
        selected_strategies = self.strategy_selector.select_strategies(
            market_type=context.market_type,
            regime=context.regime.value,
            session=context.session or self.market.get_session_name(timestamp),
            volatility=context.volatility.value,
            top_n=3,
        )

        relax_entry = bool(context.additional_info and context.additional_info.get("relax_entry"))
        
        if not selected_strategies:
            self.logger.warning("No strategies selected for current context")
            selected_strategy_names = list(self.available_strategies.keys())[:1]  # Fallback
        else:
            selected_strategy_names = [s.strategy_name for s in selected_strategies]
        
        # === STEP 3: Generate signals from selected strategies ===
        best_signal = None
        best_confidence = 0.0
        
        for strategy_name in selected_strategy_names:
            if strategy_name not in self.available_strategies:
                self.logger.warning(f"Strategy {strategy_name} not available")
                continue
            
            strategy = self.available_strategies[strategy_name]
            signal = strategy.execute_signal_generation(context)
            
            if signal and signal.direction != "HOLD" and signal.confidence > best_confidence:
                best_signal = signal
                best_confidence = signal.confidence
        
        if not best_signal or best_signal.direction == "HOLD":
            if relax_entry:
                fallback_name = selected_strategy_names[0] if selected_strategy_names else "relaxed_entry"
                best_signal = Signal(
                    direction="BUY",
                    confidence=0.1,
                    strategy_name=fallback_name,
                    reasoning="Relaxed entry mode: forcing BUY for end-to-end flow",
                )
            else:
                self.logger.info("No actionable signal generated")
                return None
        
        self.last_signals[symbol] = best_signal
        
        # === STEP 4: Check if position exists ===
        existing_position = self.broker.get_position(symbol)
        
        if existing_position:
            # Already have a position, check for exit
            exit_summary = self._check_exit_conditions(
                existing_position, context, best_signal
            )
            if exit_summary:
                return exit_summary
            
            self.logger.info(f"Existing {existing_position.direction} position held")
            return None
        
        # === STEP 5: Validate risk and position size ===
        can_trade, risk_reason = self.risk_manager.can_open_trade(
            current_open_trades=len(self.broker.get_open_positions()),
            account_balance=self.broker.current_balance,
            daily_loss=0.0,  # Would be tracked separately
        )
        
        if not can_trade:
            self.logger.warning(f"Risk check failed: {risk_reason}")
            return None
        
        # === STEP 6: Determine SL/TP based on market rules ===
        risk_multipliers = self.market.get_risk_multipliers(context.regime.value)
        
        # ATR from context (must be in indicators)
        atr = context.current_bar.get("ATR", 1.0)
        
        entry_price = context.current_bar["close"]
        
        if best_signal.direction == "BUY":
            stop_loss = entry_price - (atr * risk_multipliers.sl_atr_multiple)
            take_profit = entry_price + (atr * risk_multipliers.tp_atr_multiple)
        else:  # SELL
            stop_loss = entry_price + (atr * risk_multipliers.sl_atr_multiple)
            take_profit = entry_price - (atr * risk_multipliers.tp_atr_multiple)
        
        if pd.isna(entry_price) or pd.isna(atr) or pd.isna(stop_loss):
            self.logger.warning("Invalid price/ATR values; skipping trade")
            return None

        # === STEP 7: Calculate position size ===
        position_size = self.risk_manager.calculate_position_size(
            account_balance=self.broker.current_balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
        )
        
        if position_size <= 0:
            self.logger.warning("Cannot calculate valid position size")
            return None
        
        # === STEP 8: Validate order ===
        is_valid, validation_reason = self.market.validate_order(
            symbol=symbol,
            quantity=position_size,
            price=entry_price,
        )
        
        if not is_valid:
            self.logger.warning(f"Order validation failed: {validation_reason}")
            return None
        
        # === STEP 9: Place order ===
        order = self.broker.place_order(
            symbol=symbol,
            market_type=context.market_type,
            direction=best_signal.direction,
            quantity=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=best_signal.strategy_name,
            signal_reason=best_signal.reasoning,
            regime=context.regime.value,
            volatility=context.volatility.value,
        )
        
        # === STEP 10: Simulate immediate fill (for paper trading) ===
        self.broker.fill_order(order.order_id, entry_price, position_size)
        
        # === STEP 11: Log execution ===
        execution_summary = {
            "timestamp": timestamp,
            "symbol": symbol,
            "market_type": context.market_type,
            "signal": best_signal.direction,
            "strategy": best_signal.strategy_name,
            "confidence": best_signal.confidence,
            "entry_price": entry_price,
            "position_size": position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "regime": context.regime.value,
            "volatility": context.volatility.value,
            "atr": atr,
        }
        
        self.execution_log.append(execution_summary)
        
        self.logger.info(
            f"\U0001f4b0 ENTRY: {best_signal.strategy_name} -> {best_signal.direction} "
            f"{position_size} {symbol} @ ${entry_price:.2f} | "
            f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | "
            f"Confidence: {best_signal.confidence:.2f} | "
            f"Regime: {context.regime.value} | ATR: {atr:.2f}"
        )
        
        return execution_summary

    def _check_exit_conditions(
        self,
        position,
        context: StrategyContext,
        current_signal: Signal,
    ) -> Optional[Dict]:
        """Check if position should be closed"""
        current_price = context.current_bar["close"]
        position.update_pnl(current_price)
        
        # Check SL/TP
        if position.stop_loss:
            if position.direction == "LONG" and current_price <= position.stop_loss:
                return self.broker.close_position(
                    position.symbol, current_price, "Stop Loss Hit"
                )
            elif position.direction == "SHORT" and current_price >= position.stop_loss:
                return self.broker.close_position(
                    position.symbol, current_price, "Stop Loss Hit"
                )
        
        if position.take_profit:
            if position.direction == "LONG" and current_price >= position.take_profit:
                return self.broker.close_position(
                    position.symbol, current_price, "Take Profit Hit"
                )
            elif position.direction == "SHORT" and current_price <= position.take_profit:
                return self.broker.close_position(
                    position.symbol, current_price, "Take Profit Hit"
                )
        
        # Check for opposite signal (reverse trade)
        if current_signal.direction != "HOLD" and current_signal.direction != position.entry_signal:
            return self.broker.close_position(
                position.symbol, current_price, "Reverse Signal"
            )
        
        return None

    def get_execution_log(self) -> List[Dict]:
        """Return all executions"""
        return self.execution_log

    def __str__(self) -> str:
        return (
            f"ExecutionEngine | Market: {self.market} | "
            f"Strategies: {len(self.available_strategies)} | "
            f"Executions: {len(self.execution_log)}"
        )
