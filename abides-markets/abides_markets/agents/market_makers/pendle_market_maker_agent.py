import logging
from math import floor, ceil
from typing import Dict, List, Optional, Tuple

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns, fmt_ts

from ...utils import sigmoid
from ...messages.marketdata import (
    MarketDataMsg,
    L2SubReqMsg,
    BookImbalanceDataMsg,
    BookImbalanceSubReqMsg,
    MarketDataEventMsg,
)
from ...messages.query import QuerySpreadResponseMsg, QueryTransactedVolResponseMsg
from ...orders import Side
from ..trading_agent import TradingAgent
from ..utils import tick_to_rate, rate_to_tick


logger = logging.getLogger(__name__)

INITIAL_SPREAD_VALUE = 50


class PendleMarketMakerAgent(TradingAgent):
    """This class implements a modification of the Chakraborty-Kearns `ladder` market-making strategy, wherein the
    the size of order placed at each level is set as a fraction of measured transacted volume in the previous time
    period.

    Can skew orders to size of current inventory using beta parameter, whence beta == 0 represents inventory being
    ignored and beta == infinity represents all liquidity placed on one side of book.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        pov: float = 0.025,
        min_order_size: int = 1,
        window_size: int = 5,
        num_ticks: int = 10,
        level_spacing: float = 0.5,
        poisson_arrival: bool = True,
        log_orders: bool = False,
        collateral: float = 100_000,
        min_imbalance=0.9,
        r_bar:float = 0,
        cancel_limit_delay: NanosecondTime = str_to_ns("0"),
        wake_up_freq: NanosecondTime = str_to_ns("1h")
    ) -> None:

        super().__init__(id, name, type, random_state, symbol, log_orders, collateral)
        self.symbol: str = symbol  # Symbol traded
        self.pov: float = (
            pov  # fraction of transacted volume placed at each price level
        )
        self.min_order_size: int = (
            min_order_size  # minimum size order to place at each level, if pov <= min
        )
        self.window_size: int = window_size
        self.num_ticks: int = num_ticks  # number of ticks on each side of window in which to place liquidity
        self.level_spacing: float = (
            level_spacing  #  level spacing as a fraction of the spread
        )
        self.wake_up_freq: NanosecondTime = wake_up_freq  # Frequency of agent wake up
        self.poisson_arrival: bool = (
            poisson_arrival  # Whether to arrive as a Poisson process
        )
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq

        self.min_imbalance = min_imbalance

        self.cancel_limit_delay: NanosecondTime = cancel_limit_delay  # delay in nanoseconds between order cancellations and new limit order placements

        self.log_orders: float = log_orders

        self.has_subscribed = False

        ## Internal variables
        self.state: Dict[str, bool] = self.initialise_state()
        self.buy_order_size: int = self.min_order_size
        self.sell_order_size: int = self.min_order_size

        self.last_mid: int = rate_to_tick(r_bar)  # last observed mid price

        self.tick_size: Optional[int] = (
            ceil(INITIAL_SPREAD_VALUE * self.level_spacing)
        )

    def initialise_state(self) -> Dict[str, bool]:
        """Returns variables that keep track of whether spread and transacted volume have been observed."""

        return {"AWAITING_SPREAD": True, "AWAITING_TRANSACTED_VOLUME": True}

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def kernel_stopping(self) -> None:
        # Always call parent method to be safe.
        super().kernel_stopping()

    def wakeup(self, current_time: NanosecondTime):
        """Agent wakeup is determined by self.wake_up_freq."""

        can_trade = super().wakeup(current_time)

        if not self.has_subscribed:
            super().request_data_subscription(
                BookImbalanceSubReqMsg(
                    symbol=self.symbol,
                    min_imbalance=self.min_imbalance,
                )
            )
            self.last_time_book_order = current_time
            self.has_subscribed = True

        if can_trade:
            self.cancel_all_orders()
            self.delay(self.cancel_limit_delay)
            self.get_current_spread(self.symbol)
            self.get_transacted_volume(self.symbol, lookback_period=self.wake_up_freq)
            self.initialise_state()

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """Processes message from exchange.
        Main function is to update orders in orderbook relative to mid-price.

        Arguments:
            current_time: Simulation current time.
            message: Message received by self from ExchangeAgent.
        """

        super().receive_message(current_time, sender_id, message)

        mid = None
        if self.last_mid is not None:
            mid = self.last_mid

        if (
            isinstance(message, QueryTransactedVolResponseMsg)
            and self.state["AWAITING_TRANSACTED_VOLUME"] is True
        ):
            self.update_order_size()
            self.state["AWAITING_TRANSACTED_VOLUME"] = False

        if isinstance(message, BookImbalanceDataMsg):
            if message.stage == MarketDataEventMsg.Stage.START:
                try:
                    self.logEvent("UNBALANCED_MARKET", f"{round(100*message.imbalance, 2)}% {message.side}")
                    self.place_orders(mid)
                    self.last_time_book_order = current_time
                except:
                    logger.warning("Book is imbalance and cannot place order as a market maker.")

        if (
            isinstance(message, QuerySpreadResponseMsg)
            and self.state["AWAITING_SPREAD"] is True
        ):
            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
            if bid and ask:
                mid = int((ask + bid) / 2)
                self.last_mid = mid

                self.state["AWAITING_SPREAD"] = False
            else:
                logger.info(f"SPREAD MISSING at time {fmt_ts(current_time)}")
                if bid: 
                    mid = bid
                    self.last_mid = mid
                elif ask:
                    mid = ask
                    self.last_mid = mid
                self.state[
                    "AWAITING_SPREAD"
                ] = False  # use last mid price and spread

        if (
            self.state["AWAITING_SPREAD"] is False
            and self.state["AWAITING_TRANSACTED_VOLUME"] is False
            and mid is not None
        ):
            self.place_orders(mid)
            self.state = self.initialise_state()
            self.set_wakeup(current_time + self.get_wake_frequency())

    def update_order_size(self) -> None:
        """Updates size of order to be placed."""

        buy_transacted_volume = self.transacted_volume[self.symbol][0]
        sell_transacted_volume = self.transacted_volume[self.symbol][1]
        total_transacted_volume = buy_transacted_volume + sell_transacted_volume

        # logger.info(f"{fmt_ts(self.current_time)} - Recorded transacted volume {total_transacted_volume}: {buy_transacted_volume} BUY, {sell_transacted_volume} SELL")
        qty = round(self.pov * total_transacted_volume)

        self.buy_order_size = (
            qty if qty >= self.min_order_size else self.min_order_size
        )
        self.sell_order_size = (
            qty if qty >= self.min_order_size else self.min_order_size
            )

    def compute_orders_to_place(self, mid: int) -> Tuple[List[int], List[int]]:
        """Given a mid price, computes the orders that need to be placed to
        orderbook, and adds these orders to bid and ask deques.

        Arguments:
            mid: Mid price.
        """

        mid_point = mid

        highest_bid = int(mid_point) - floor(0.5 * self.window_size)
        lowest_ask = int(mid_point) + ceil(0.5 * self.window_size)

        lowest_bid = highest_bid - ((self.num_ticks - 1) * self.tick_size)
        highest_ask = lowest_ask + ((self.num_ticks - 1) * self.tick_size)

        bids_to_place = [
            price
            for price in range(lowest_bid, highest_bid + self.tick_size, self.tick_size)
        ]
        asks_to_place = [
            price
            for price in range(lowest_ask, highest_ask + self.tick_size, self.tick_size)
        ]

        return bids_to_place, asks_to_place

    def place_orders(self, mid: int) -> None:
        """Given a mid-price, compute new orders that need to be placed, then
        send the orders to the Exchange.

        Arguments:
            mid: Mid price.
        """

        bid_orders, ask_orders = self.compute_orders_to_place(mid)

        orders = []

        for i, bid_price in enumerate(bid_orders):
            logger.debug(
                "{}: Placing BUY limit order of size {} @ price {}",
                self.name,
                self.buy_order_size,
                bid_price,
            )
            orders.append(
                self.create_limit_order(
                    self.symbol, self.buy_order_size*(10-i)/45*20, Side.BID, bid_price
                )
            )

        for i, ask_price in enumerate(ask_orders):
            logger.debug(
                "{}: Placing SELL limit order of size {} @ price {}",
                self.name,
                self.sell_order_size,
                ask_price,
            )
            orders.append(
                self.create_limit_order(
                    self.symbol, self.sell_order_size**(10-i)/45*20, Side.ASK, ask_price
                )
            )

        self.place_multiple_orders(orders)

    def get_wake_frequency(self) -> NanosecondTime:
        if not self.poisson_arrival:
            return self.wake_up_freq
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
            return int(round(delta_time))