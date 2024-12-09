import datetime as dt
import logging
from math import sqrt
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from ..agents.utils import tick_to_rate, rate_to_tick

from .oracle import Oracle


logger = logging.getLogger(__name__)


class ManualOracle(Oracle):
    def __init__(
        self,
        mkt_open: NanosecondTime,
        mkt_close: NanosecondTime,
        symbols: Dict[str, Dict[str, Any]],
        megashock: List[Dict[str, float]] = []
    ) -> None:
        self.mkt_open: NanosecondTime = mkt_open
        self.mkt_close: NanosecondTime = mkt_close
        self.symbols: Dict[str, Dict[str, Any]] = symbols
        self.megashock: List[Dict[str, float]] = megashock

        self.freq: str = "1min"

        # The dictionary r holds the fundamenal value series for each symbol.
        self.r: Dict[str, np.array] = {}

        for symbol in symbols:
            s = symbols[symbol]
            self.r[symbol] = self.generate_fundamental_value_series(symbol=symbol, **s)

    def generate_fundamental_value_series(
        self, symbol: str, r_bar: float, kappa: float, sigma_s: float
    ) -> np.array:
        """Generates the fundamental value series for a single stock symbol.

        Arguments:
            symbol: The symbold to calculate the fundamental value series for.
            r_bar: The mean fundamental value.
            kappa: The mean reversion coefficient.
            sigma_s: The shock variance.  (Note: NOT STANDARD DEVIATION)

        Because the oracle uses the global np.random PRNG to create the
        fundamental value series, it is important to create the oracle BEFORE
        the agents.  In this way the addition of a new agent will not affect the
        sequence created.  (Observations using the oracle will use an agent's
        PRNG and thus not cause a problem.)
        """

        # Turn variance into std.
        sigma_s = sqrt(sigma_s)

        # Create the time series into which values will be projected and initialize the first value.
        date_range = pd.date_range(
            self.mkt_open, self.mkt_close, freq=self.freq
        )

        s = pd.Series(index=date_range)
        r = np.zeros(len(s.index))
        r_bar_tick = rate_to_tick(r_bar)
        r[0] = r_bar_tick

        # Predetermine the random shocks for all time steps (at once, for computation speed).
        shock = np.random.normal(scale=sigma_s, size=(r.shape[0]))
    
        for megashock in self.megashock:
            index = int(megashock["time"]*r.shape[0])
            shock[index] = megashock["mag"]

        # Compute the value series.
        for t in range(1, r.shape[0]):
            r[t] = max(0, (kappa * r_bar_tick) + ((1 - kappa) * r[t - 1]) + shock[t])

        # Replace the series values with the fundamental value series.  Round and convert to
        # integer cents.
        r = np.round(r)

        return r.astype(int)

    def get_daily_open_price(
        self, symbol: str, mkt_open: NanosecondTime, cents: bool = True
    ) -> int:
        """Return the daily open price for the symbol given.

        In the case of the MeanRevertingOracle, this will simply be the first
        fundamental value, which is also the fundamental mean. We will use the
        mkt_open time as given, however, even if it disagrees with this.
        """

        # If we did not already know mkt_open, we should remember it.
        if (mkt_open is not None) and (self.mkt_open is None):
            self.mkt_open = mkt_open

        logger.debug(
            "Oracle: client requested {symbol} at market open: {}", self.mkt_open
        )

        open_price = self.r[symbol].loc[self.mkt_open]
        logger.debug("Oracle: market open price was was {}", open_price)

        return open_price

    def observe_price(
        self,
        symbol: str,
        current_time: NanosecondTime,
        random_state: np.random.RandomState,
        sigma_n: int = 50**2  # sd of 50 ticks
    ) -> int:
        """Return a noisy observation of the current fundamental value (NOTE: in term of tick).

        While the fundamental value for a given equity at a given time step does
        not change, multiple agents observing that value will receive different
        observations.

        Only the Exchange or other privileged agents should use noisy=False.

        sigma_n is experimental observation variance.  NOTE: NOT STANDARD DEVIATION.

        Each agent must pass its RandomState object to ``observe_price``.  This
        ensures that each agent will receive the same answers across multiple
        same-seed simulations even if a new agent has been added to the experiment.
        """

        # If the request is made after market close, return the close price.
        if current_time >= self.mkt_close:
            r_t = self.r[symbol][-1]
        else:
            last_time = int((current_time-self.mkt_open)/str_to_ns(self.freq))
            r_t = self.r[symbol][last_time]

        # Generate a noisy observation of fundamental value at the current time.
        if sigma_n == 0:
            obs = r_t
        else:
            obs = int(round(random_state.normal(loc=r_t, scale=sqrt(sigma_n))))

        # Reminder: all simulator prices are specified in integer tick.
        return obs
