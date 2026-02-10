"""
Strategy Registry - Dynamic Strategy Discovery and Management

Auto-discovers all strategies under TradingPlatform/strategies/generic
and provides filtering, querying, and instantiation capabilities.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type

from TradingPlatform.core.base_strategy import BaseStrategy, MarketRegime, VolatilityBucket

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Singleton registry for auto-discovering and managing strategies.

    Automatically scans the strategies/generic directory for classes that
    inherit from BaseStrategy.
    """

    _instance = None
    _strategies: Dict[str, Type[BaseStrategy]] = {}
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def _discover_strategies(cls) -> None:
        """
        Auto-discover all strategy classes in the strategies/generic folder.
        """
        if cls._initialized:
            return

        strategies_dir = Path(__file__).parent / "generic"
        logger.info("Discovering strategies in: %s", strategies_dir)

        py_files = strategies_dir.glob("*.py")
        excluded_files = {"__init__.py"}

        for py_file in py_files:
            if py_file.name in excluded_files:
                continue

            module_name = py_file.stem
            module_path = f"TradingPlatform.strategies.generic.{module_name}"

            try:
                module = importlib.import_module(module_path)

                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseStrategy)
                        and obj is not BaseStrategy
                        and obj.__module__ == module_path
                    ):
                        cls._strategies[obj.__name__] = obj
                        logger.info("Discovered strategy: %s", obj.__name__)
            except Exception as exc:
                logger.error("Error loading module %s: %s", module_path, exc, exc_info=True)

        cls._initialized = True
        logger.info("Strategy discovery complete. Found %d strategies.", len(cls._strategies))

    @classmethod
    def get_all(cls) -> Dict[str, Type[BaseStrategy]]:
        cls._discover_strategies()
        return cls._strategies.copy()

    @classmethod
    def get(cls, strategy_name: str) -> Optional[Type[BaseStrategy]]:
        cls._discover_strategies()
        return cls._strategies.get(strategy_name)

    @classmethod
    def list_names(cls) -> List[str]:
        cls._discover_strategies()
        return sorted(cls._strategies.keys())

    @classmethod
    def get_supported(
        cls,
        market: Optional[str] = None,
        timeframe: Optional[str] = None,
        regime: Optional[MarketRegime] = None,
        volatility: Optional[VolatilityBucket] = None,
    ) -> List[Type[BaseStrategy]]:
        cls._discover_strategies()

        matched_strategies: List[Type[BaseStrategy]] = []

        for _, strategy_class in cls._strategies.items():
            try:
                temp_instance = strategy_class()

                if market and not temp_instance.supports_market(market):
                    continue

                if regime and not temp_instance.supports_regime(regime):
                    continue

                if volatility and not temp_instance.supports_volatility(volatility):
                    continue

                matched_strategies.append(strategy_class)
            except Exception as exc:
                logger.warning(
                    "Error checking support for %s: %s",
                    strategy_class.__name__,
                    exc,
                    exc_info=False,
                )
                continue

        return matched_strategies

    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Optional[Dict[str, object]]:
        strategy_class = cls.get(strategy_name)
        if not strategy_class:
            return None

        try:
            temp_instance = strategy_class()
            return {
                "name": temp_instance.name,
                "version": temp_instance.version,
                "enabled": temp_instance.enabled,
                "required_indicators": temp_instance.required_indicators(),
                "class_name": strategy_class.__name__,
                "module": strategy_class.__module__,
                "docstring": strategy_class.__doc__,
            }
        except Exception as exc:
            logger.error("Error getting info for %s: %s", strategy_name, exc)
            return {"error": str(exc)}

    @classmethod
    def reload(cls) -> None:
        cls._strategies.clear()
        cls._initialized = False
        cls._discover_strategies()
        logger.info("Strategy registry reloaded.")

    @classmethod
    def summary(cls) -> str:
        cls._discover_strategies()

        lines = [
            "=" * 60,
            "STRATEGY REGISTRY SUMMARY",
            "=" * 60,
            f"Total Strategies: {len(cls._strategies)}",
            "",
            "Available Strategies:",
            "-" * 60,
        ]

        for name in sorted(cls._strategies.keys()):
            info = cls.get_strategy_info(name)
            if info:
                lines.append(f"  • {name} (v{info.get('version', 'unknown')})")
                if info.get("docstring"):
                    first_line = info["docstring"].split("\n")[0].strip()
                    lines.append(f"      {first_line}")

        lines.append("=" * 60)
        return "\n".join(lines)


def get_strategy(name: str) -> Optional[Type[BaseStrategy]]:
    return StrategyRegistry.get(name)


__all__ = ["StrategyRegistry", "get_strategy"]
