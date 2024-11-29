import logging
from typing import Optional

import numpy as np

from abides_core import Message, NanosecondTime

from ..generators import OrderSizeGenerator
from ..messages.query import QuerySpreadResponseMsg
from ..orders import Side
from .trading_agent import TradingAgent


logger = logging.getLogger(__name__)


class PendleSeedingAgent(TradingAgent):
    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        symbol: str = "PEN",
        log_orders: float = False,
        collateral: float = 100_000,
        size: int = 100,
        min_bid: int = 1,
        max_bid: int = 10,
        min_ask: int = 11,
        max_ask: int = 20
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state, symbol, log_orders, collateral)
        self.symbol = symbol
        
        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state: str = "AWAITING_WAKEUP"

        # Params for seeding process
        self.size: int = size
        self.min_bid: int = min_bid
        self.max_bid: int = max_bid
        self.min_ask: int = min_ask
        self.max_ask: int = max_ask
        
        
    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # self.kernel is set in Agent.kernel_initializing()
        # self.exchange_id is set in TradingAgent.kernel_starting()

        super().kernel_starting(start_time)

        self.rate_oracle = self.kernel.rate_oracle

    def kernel_stopping(self) -> None:
        # Always call parent method to be safe.
        super().kernel_stopping()

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(current_time)

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return

        # Steady state wakeup behavior starts here.
        # If we've been told the market has closed, we stop.
        if self.mkt_closed:
            return
        
        if type(self) == PendleSeedingAgent:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
        else:
            self.state = "AWAITING_WAKEUP"

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receive_message(current_time, sender_id, message)

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        if self.state == "AWAITING_SPREAD":
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.

            if isinstance(message, QuerySpreadResponseMsg):
                # This is what we were waiting for.

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed:
                    return

                bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
                if not bid and not ask:  # The orderbook is blank, need seeding
                    self.seed()
                    self.state = "AWAITING_WAKEUP"

    def seed(self) -> None:
        # Seeding
        self.logEvent("SEEDING")
        
        for price in range(self.min_bid, self.max_bid+1):
            self.place_limit_order(self.symbol, self.size, Side.BID, price)
        for price in range(self.min_ask, self.max_ask+1):
            self.place_limit_order(self.symbol, self.size, Side.ASK, price)

    # For future use
    def get_wake_frequency(self) -> NanosecondTime:
        return self.random_state.randint(low=0, high=100)
