"""Brain-chassis interface for the trading framework.

Any trading strategy ("brain") implements the Strategy ABC and returns a list
of TradeIntent objects. The execution chassis acts only on this interface, never
on a concrete strategy — so swapping strategies means adding one new Strategy
subclass and a config string, with no import changes in the chassis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Action(Enum):
    # A brain emits OPEN to enter a position or CLOSE to exit one it holds.
    # Positions the brain does NOT name are left untouched (silence = hold),
    # which minimizes Alpaca transactions and slippage.
    OPEN = "open"
    CLOSE = "close"


@dataclass
class TradeIntent:
    # One desired position change from a brain. Only fields the chassis ACTS on
    # are first-class; anything brain-specific goes in metadata (logged, not acted on).
    action: Action
    asset: str                            # ticker the chassis routes orders for
    target_weight: float | None = None    # signed desired weight; required for OPEN, ignored for CLOSE
    stop_level: float | None = None        # absolute price stop; chassis exits if breached (strategy-agnostic)
    horizon_bars: int | None = None        # optional time stop; chassis exits after N bars
    metadata: dict = field(default_factory=dict)  # brain-specific extras (e.g. confidence, source); chassis logs only


class Strategy(ABC):
    # The contract every brain implements. The chassis depends on THIS, not on
    # any concrete brain, so brains are swappable without touching chassis code.

    @property
    @abstractmethod
    def name(self) -> str:
        # Human-readable identifier, used in logging and to namespace artifacts.
        ...

    @property
    @abstractmethod
    def frequency(self) -> str:
        # Evaluation cadence the chassis schedules on: "daily", "1h", etc.
        # Designs-in per-strategy frequency (resolves the hardcoded-78 coupling).
        ...

    @property
    @abstractmethod
    def required_history(self) -> int:
        # Number of bars of history the chassis must feed generate_targets.
        ...

    @property
    @abstractmethod
    def model_dir(self) -> str:
        # Directory under the_models/ holding THIS strategy's artifacts
        # (e.g. "the_models/cointegration"). Keeps each brain's models namespaced
        # so the chassis loads the right model for the chosen strategy.
        ...

    @abstractmethod
    def generate_targets(self, market_data: pd.DataFrame, positions) -> list[TradeIntent]:
        # Given recent market_data (price panel) and current positions, return the
        # desired position CHANGES as TradeIntents. Emit intents ONLY for changes;
        # any position not named is left as-is. Sizing/risk distribution (e.g. HRP)
        # is the chassis's job, not the brain's.
        #
        # positions: the chassis's current-holdings representation (loosely typed
        # for now; will be tightened once the chassis position model is settled).
        ...
