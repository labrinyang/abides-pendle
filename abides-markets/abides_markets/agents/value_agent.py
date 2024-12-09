import logging
from typing import Optional, List

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from ..generators import OrderSizeGenerator
from ..messages.query import QuerySpreadResponseMsg
from ..orders import Side
from .trading_agent import TradingAgent
from .utils import tick_to_rate, rate_to_tick


logger = logging.getLogger(__name__)


class ValueAgent(TradingAgent):
    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        symbol: str = "PEN",
        log_orders: float = False,
        collateral: float = 100_000,
        order_size_model: Optional[OrderSizeGenerator] = None,
        r_bar: float = 0.10,
        wake_up_freq: NanosecondTime = str_to_ns("10min"),
        coef: List[float] = [0.05, 0.40]
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state, symbol, log_orders, collateral)

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading: bool = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state: str = "AWAITING_WAKEUP"

        # The agent maintains two priors: r_t and sigma_t (value and error estimates).
        self.r_t: float = r_bar
        self.wake_up_freq: NanosecondTime = wake_up_freq

        self.order_size_model = order_size_model  # Probabilistic model for order size

        self.funding_rate_coef: float = coef[0]
        self.oracle_coef: float = coef[1]
        assert coef[0] >= 0 and coef[0] < 1
        assert coef[1] >= 0 and coef[1] < 1
        assert sum(coef) < 1

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # self.kernel is set in Agent.kernel_initializing()
        # self.rate_oracle is set in Agent.kernel_initializing()
        # self.exchange_id is set in TradingAgent.kernel_starting()

        super().kernel_starting(start_time)

        self.driving_oracle = self.kernel.driving_oracle

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
                logger.debug("%s is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        self.set_wakeup(current_time + self.get_wake_frequency())

        if self.mkt_closed and (not self.symbol in self.daily_close_price):
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        self.cancel_all_orders()

        if type(self) == ValueAgent:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
        else:
            self.state = "ACTIVE"

    def updateEstimates(self) -> int:
        # Naive approach
        obs_t = self.driving_oracle.observe_price(
            self.symbol,
            self.current_time,
            random_state=self.random_state,
        )
        last_funding_rate = self.rate_oracle.get_floating_rate(self.current_time)/self.kernel.rate_normalizer

        self.r_t = (1 - self.oracle_coef - self.funding_rate_coef)*self.r_t + self.funding_rate_coef*last_funding_rate + self.oracle_coef*tick_to_rate(obs_t)
        # self.logEvent("NEW ESTIMATE", self.r_t)

        return self.r_t

    def placeOrder(self) -> None:
        # estimate final value of the fundamental price
        # used for surplus calculation
        r_t = self.updateEstimates()

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)

        if bid and ask:
            mid = int((ask + bid) / 2)
            spread = abs(ask - bid)

            if r_t < tick_to_rate(mid):
                # fundamental belief that price will go down, place a sell order
                buy = False
                p = (
                    bid
                )  # submit a market order to sell, limit order inside the spread or deeper in the book
            elif r_t >= tick_to_rate(mid):
                # fundamental belief that price will go up, buy order
                buy = True
                p = (
                    ask
                )  # submit a market order to buy, a limit order inside the spread or deeper in the book
        else:
            # initialize randomly
            buy = self.random_state.randint(0, 1 + 1)
            p = rate_to_tick(r_t)

        # Place the order
        percentage = self.order_size_model.sample(random_state=self.random_state)
        MtM = self.mark_to_market()
        size = int(percentage*MtM/(0.05*self.n_payment*self.rate_normalizer))

        side = Side.BID if buy == 1 else Side.ASK

        if size > 0:
            self.place_limit_order(self.symbol, size, side, p)

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
                self.placeOrder()
                self.state = "AWAITING_WAKEUP"

        # Cancel all open orders.
        # Return value: did we issue any cancellation requests?

    def get_wake_frequency(self) -> NanosecondTime:
        return int(round(self.random_state.exponential(self.wake_up_freq)))