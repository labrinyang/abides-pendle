import logging
from typing import Optional

import numpy as np

from abides_core import Message, NanosecondTime

from ..generators import OrderSizeGenerator
from ..messages.query import QuerySpreadResponseMsg
from ..orders import Side
from .trading_agent import TradingAgent


logger = logging.getLogger(__name__)


class NoiseAgent(TradingAgent):
    """
    Noise agent implement simple strategy. The agent wakes up once and places 1 order.
    """

    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        symbol: str = "PEN",
        log_orders: bool = False,
        collateral: int = 100000,
        order_size_model: Optional[OrderSizeGenerator] = None,
        wakeup_time: Optional[NanosecondTime] = None,
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state, symbol, log_orders, collateral)

        self.wakeup_time: NanosecondTime = wakeup_time

        self.symbol: str = symbol  # symbol to trade

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading: bool = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state: str = "AWAITING_WAKEUP"

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time: Optional[NanosecondTime] = None

        self.order_size_model = order_size_model  # Probabilistic model for order size

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # self.kernel is set in Agent.kernel_initializing()
        # self.exchange_id is set in TradingAgent.kernel_starting()

        super().kernel_starting(start_time)

        # self.oracle = self.kernel.oracle

    def kernel_stopping(self) -> None:
        # Always call parent method to be safe.
        super().kernel_stopping()

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(current_time)

        self.state = "INACTIVE"

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True

                # Time to start trading!
                logger.debug("{} is ready to start trading now.".format(self.name))


        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        if self.wakeup_time > current_time:
            self.set_wakeup(self.wakeup_time)
            return

        if self.mkt_closed and (self.symbol not in self.daily_close_price):
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        if type(self) == NoiseAgent:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
        else:
            self.state = "ACTIVE"

    def placeOrder(self) -> None:
        # place order in random direction at a mid
        buy_indicator = self.random_state.randint(0, 1 + 1)

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)

        percentage = self.order_size_model.sample(random_state=self.random_state)
        MtM = self.mark_to_market()
        size = int(percentage*MtM/(0.05*self.n_payment*self.rate_normalizer))

        if size > 0:
            if buy_indicator == 1 and ask:
                self.place_limit_order(self.symbol, size, Side.BID, ask)
            elif not buy_indicator and bid:
                self.place_limit_order(self.symbol, size, Side.ASK, bid)

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

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                # logger.info(f"HEREEEEEE {self.get_known_bid_ask_midpoint(self.symbol)}")
                self.placeOrder()
                self.state = "AWAITING_WAKEUP"

    # Internal state and logic specific to this agent subclass.

    def get_wake_frequency(self) -> NanosecondTime:
        return self.random_state.randint(low=0, high=100)
