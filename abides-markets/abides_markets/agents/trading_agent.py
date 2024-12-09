import logging
import sys
import warnings
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import fmt_ts, merge_swap, str_to_ns

from ..messages.market import (
    MarketClosePriceRequestMsg,
    MarketClosePriceMsg,
    MarketClosedMsg,
    MarketHoursRequestMsg,
    MarketHoursMsg,
)
from ..messages.marketdata import MarketDataSubReqMsg, MarketDataMsg, L2DataMsg
from ..messages.order import (
    LimitOrderMsg,
    MarketOrderMsg,
    PartialCancelOrderMsg,
    CancelOrderMsg,
    ModifyOrderMsg,
    ReplaceOrderMsg,
)
from ..messages.orderbook import (
    OrderAcceptedMsg,
    OrderExecutedMsg,
    OrderCancelledMsg,
    OrderPartialCancelledMsg,
    OrderModifiedMsg,
    OrderReplacedMsg,
)
from ..messages.query import (
    QueryLastTradeMsg,
    QueryLastTradeResponseMsg,
    QuerySpreadMsg,
    QuerySpreadResponseMsg,
    QueryOrderStreamMsg,
    QueryOrderStreamResponseMsg,
    QueryTransactedVolMsg,
    QueryTransactedVolResponseMsg,
)
from ..orders import Order, LimitOrder, MarketOrder, Side
from .financial_agent import FinancialAgent
from .exchange_agent import ExchangeAgent
from .utils import tick_to_rate, rate_to_tick


logger = logging.getLogger(__name__)


class TradingAgent(FinancialAgent):
    """
    The TradingAgent class (via FinancialAgent, via Agent) is intended as the
    base class for all trading agents (i.e. not things like exchanges) in a
    market simulation.

    It handles a lot of messaging (inbound and outbound) and state maintenance
    automatically, so subclasses can focus just on implementing a strategy without
    too much bookkeeping.
    """

    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        symbol: str = "PEN",
        log_orders: bool = False,
        collateral: float = 100_000,
        pos_size: float = 0,
        pos_fixrate: float = 0,
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state)

        # We don't yet know when the exchange opens or closes.
        self.mkt_open: Optional[NanosecondTime] = None
        self.mkt_close: Optional[NanosecondTime] = None

        self.symbol = symbol

        # Log order activity?
        self.log_orders: bool = log_orders

        # Log all activity to file?
        if log_orders is None:
            self.log_orders = False
            self.log_to_file = False

        self.starting_collateral: float = collateral

        # PENDLE
        self.position: Dict[str, float] = {"COLLATERAL": collateral,
                                         "SIZE": pos_size,
                                         "FIXRATE":pos_fixrate}
        self.pen_orders: Dict[int, Order] = {}
        # END PENDLE

        # TradingAgent has constants to support simulated market orders.
        self.MKT_BUY = sys.maxsize
        self.MKT_SELL = 0

        # The base TradingAgent will track its position and outstanding orders.
        # Orders is a dictionary of active, open orders
        # (not cancelled, not fully executed) keyed by order_id.
        self.orders: Dict[int, Order] = {}

        # The base TradingAgent also tracks last known prices for every symbol
        # for which it has received as QUERY_LAST_TRADE message.  Subclass
        # agents may use or ignore this as they wish.  Note that the subclass
        # agent must request pricing when it wants it.  This agent does NOT
        # automatically generate such requests, though it has a helper function
        # that can be used to make it happen.
        self.last_trade: Dict[str, int] = {}

        self.last_twap: Optional[float] = None

        # used in subscription mode to record the timestamp for which the data was current in the ExchangeAgent
        self.exchange_ts: Dict[str, NanosecondTime] = {}

        # When a last trade price comes in after market close, the trading agent
        # automatically records it as the daily close price for a symbol.
        self.daily_close_price: Dict[str, int] = {}

        self.nav_diff: int = 0
        self.basket_size: int = 0

        # The agent remembers the last known bids and asks (with variable depth,
        # showing only aggregate volume at each price level) when it receives
        # a response to QUERY_SPREAD.
        self.known_bids: Dict = {}
        self.known_asks: Dict = {}

        # The agent remembers the order history communicated by the exchange
        # when such is requested by an agent (for example, a heuristic belief
        # learning agent).
        self.stream_history: Dict[str, Any] = {}

        # The agent records the total transacted volume in the exchange for a given symbol and lookback period
        self.transacted_volume: Dict = {}

        # Each agent can choose to log the orders executed
        self.executed_orders: List = []

        # For special logging at the first moment the simulator kernel begins
        # running (which is well after agent init), it is useful to keep a simple
        # boolean flag.
        self.first_wake: bool = True

        # Remember whether we have already passed the exchange close time, as far
        # as we know.
        self.mkt_closed: bool = False

        self.last_R1 = np.inf
        self.last_R2 = np.inf
        self.last_mm = 0
        self.n_payment: Optional[int] = None
        self.rate_normalizer: Optional[float] = None

    # Simulation lifecycle messages.

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        """
        Arguments:
            start_time: The time that the simulation started.
        """

        assert self.kernel is not None

        # self.kernel is set in Agent.kernel_initializing()
        self.logEvent("STARTING_COLLATERAL", self.starting_collateral, True)

        # Find an exchange with which we can place orders.  It is guaranteed
        # to exist by now (if there is one).
        self.exchange_id: int = self.kernel.find_agents_by_type(ExchangeAgent)[0]
        self.rate_normalizer = self.kernel.rate_normalizer

        logger.debug(
            f"Agent {self.id} requested agent of type Agent.ExchangeAgent.  Given Agent ID: {self.exchange_id}"
        )

        # Request a wake-up call as in the base Agent.
        super().kernel_starting(start_time)

    def kernel_stopping(self) -> None:
        # Always call parent method to be safe.
        super().kernel_stopping()

        assert self.kernel is not None


        # Record final results for presentation/debugging.  This is an ugly way
        # to do this, but it is useful for now.
        mytype = self.type
        gain = self.position["COLLATERAL"] - self.starting_collateral
        self.logEvent("FINAL_PNL", f"{100*gain/self.starting_collateral}%", True)

        if mytype in self.kernel.mean_result_by_agent_type:
            self.kernel.mean_result_by_agent_type[mytype] += gain
            self.kernel.agent_count_by_type[mytype] += 1
        else:
            self.kernel.mean_result_by_agent_type[mytype] = gain
            self.kernel.agent_count_by_type[mytype] = 1

    # Simulation participation messages.

    def wakeup(self, current_time: NanosecondTime) -> bool:
        """
        Arguments:
            current_time: The time that this agent was woken up by the kernel.

        Returns:
            For the sake of subclasses, TradingAgent now returns a boolean
            indicating whether the agent is "ready to trade" -- has it received
            the market open and closed times, and is the market not already closed.
        """

        super().wakeup(current_time)

        if self.first_wake:
            self.first_wake = False

            # Tell the exchange we want to be sent the final prices when the market closes.
            self.send_message(self.exchange_id, MarketClosePriceRequestMsg())

        if self.mkt_open is None:
            # Ask our exchange when it opens and closes.
            self.send_message(self.exchange_id, MarketHoursRequestMsg())

        return (self.mkt_open and self.mkt_close) and not self.mkt_closed

    def request_data_subscription(
        self, subscription_message: MarketDataSubReqMsg
    ) -> None:
        """
        Used by any Trading Agent subclass to create a subscription to market data from
        the Exchange Agent.

        Arguments:
            subscription_message: An instance of a MarketDataSubReqMessage.
        """

        subscription_message.cancel = False

        self.send_message(recipient_id=self.exchange_id, message=subscription_message)

    def cancel_data_subscription(
        self, subscription_message: MarketDataSubReqMsg
    ) -> None:
        """
        Used by any Trading Agent subclass to cancel subscription to market data from
        the Exchange Agent.

        Arguments:
            subscription_message: An instance of a MarketDataSubReqMessage.
        """

        subscription_message.cancel = True

        self.send_message(recipient_id=self.exchange_id, message=subscription_message)

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """
        Arguments:
            current_time: The time that this agent received the message.
            sender_id: The ID of the agent who sent the message.
            message: The message contents.
        """

        assert self.kernel is not None

        super().receive_message(current_time, sender_id, message)

        # Do we know the market hours?
        had_mkt_hours = self.mkt_open is not None and self.mkt_close is not None

        # Record market open or close times.
        if isinstance(message, MarketHoursMsg):
            self.mkt_open = message.mkt_open
            self.mkt_close = message.mkt_close
            self.n_payment = np.floor((self.mkt_close - self.mkt_open)/self.kernel.swap_interval)

            logger.debug("Recorded market open: {}".format(fmt_ts(self.mkt_open)))
            logger.debug("Recorded market close: {}".format(fmt_ts(self.mkt_close)))

        elif isinstance(message, MarketClosePriceMsg):
            # Update our local last trade prices with the accurate last trade prices from
            # the exchange so we can accurately calculate our mark-to-market values.
            for symbol, close_price in message.close_prices.items():
                self.last_trade[symbol] = close_price

        elif isinstance(message, MarketClosedMsg):
            # We've tried to ask the exchange for something after it closed.  Remember this
            # so we stop asking for things that can't happen.
            self.market_closed()

        elif isinstance(message, OrderExecutedMsg):
            # Call the order_executed method, which subclasses should extend.  This parent
            # class could implement default "portfolio tracking" or "returns tracking"
            # behavior.
            self.order_executed(message.order)

        elif isinstance(message, OrderAcceptedMsg):
            # Call the order_accepted method, which subclasses should extend.
            self.order_accepted(message.order)

        elif isinstance(message, OrderCancelledMsg):
            # Call the order_cancelled method, which subclasses should extend.
            self.order_cancelled(message.order)

        elif isinstance(message, OrderPartialCancelledMsg):
            # Call the order_cancelled method, which subclasses should extend.
            self.order_partial_cancelled(message.new_order)

        elif isinstance(message, OrderModifiedMsg):
            # Call the order_cancelled method, which subclasses should extend.
            self.order_modified(message.new_order)

        elif isinstance(message, OrderReplacedMsg):
            # Call the order_cancelled method, which subclasses should extend.
            self.order_replaced(message.old_order, message.new_order)

        elif isinstance(message, QueryLastTradeResponseMsg):
            # Call the query_last_trade method, which subclasses may extend.
            # Also note if the market is closed.
            if message.mkt_closed:
                self.mkt_closed = True

            self.query_last_trade(message.symbol, message.last_trade)

        elif isinstance(message, QuerySpreadResponseMsg):
            # Call the query_spread method, which subclasses may extend.
            # Also note if the market is closed.
            if message.mkt_closed:
                self.mkt_closed = True

            self.query_spread(
                message.symbol, message.last_trade, message.bids, message.asks, ""
            )

        elif isinstance(message, QueryOrderStreamResponseMsg):
            # Call the query_order_stream method, which subclasses may extend.
            # Also note if the market is closed.
            if message.mkt_closed:
                self.mkt_closed = True

            self.query_order_stream(message.symbol, message.orders)

        elif isinstance(message, QueryTransactedVolResponseMsg):
            if message.mkt_closed:
                self.mkt_closed = True

            self.query_transacted_volume(
                message.symbol, message.bid_volume, message.ask_volume
            )

        elif isinstance(message, MarketDataMsg):
            self.handle_market_data(message)

        # Now do we know the market hours?
        have_mkt_hours = self.mkt_open is not None and self.mkt_close is not None

        # Once we know the market open and close times, schedule a wakeup call for market open.
        # Only do this once, when we first have both items.
        if have_mkt_hours and not had_mkt_hours:
            # Agents are asked to generate a wake offset from the market open time.  We structure
            # this as a subclass request so each agent can supply an appropriate offset relative
            # to its trading frequency.
            ns_offset = self.get_wake_frequency()

            self.set_wakeup(self.mkt_open + ns_offset)

    def get_last_trade(self, symbol: str) -> None:
        """
        Used by any Trading Agent subclass to query the last trade price for a symbol.

        This activity is not logged.

        Arguments:
            symbol: The symbol to query.
        """

        self.send_message(self.exchange_id, QueryLastTradeMsg(symbol))

    def get_current_spread(self, symbol: str, depth: int = 1) -> None:
        """
        Used by any Trading Agent subclass to query the current spread for a symbol.

        This activity is not logged.

        Arguments:
            symbol: The symbol to query.
            depth:
        """

        self.send_message(self.exchange_id, QuerySpreadMsg(symbol, depth))

    def get_order_stream(self, symbol: str, length: int = 1) -> None:
        """
        Used by any Trading Agent subclass to query the recent order s  tream for a symbol.

        Arguments:
            symbol: The symbol to query.
            length:
        """

        self.send_message(self.exchange_id, QueryOrderStreamMsg(symbol, length))

    def get_transacted_volume(
        self, symbol: str, lookback_period: NanosecondTime = str_to_ns("10min")
    ) -> None:
        """
        Used by any trading agent subclass to query the total transacted volume in a
        given lookback period.

        Arguments:
            symbol: The symbol to query.
            lookback_period: The length of time to consider when calculating the volume.
        """

        self.send_message(
            self.exchange_id, QueryTransactedVolMsg(symbol, lookback_period)
        )

    def create_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        limit_price: int,
        order_id: Optional[int] = None,
        is_hidden: bool = False,
        is_price_to_comply: bool = False,
        insert_by_id: bool = False,
        is_post_only: bool = False,
        tag: Any = None,
    ) -> LimitOrder:
        """
        Used by any Trading Agent subclass to create a limit order. Check open margin here.

        Arguments:
            symbol: A valid symbol.
            quantity: Positive share quantity.
            side: Side.BID or Side.ASK.
            limit_price: Price in cents.
            order_id: An optional order id (otherwise global autoincrement is used).
            is_hidden:
            is_price_to_comply:
            insert_by_id:
            is_post_only:
            ignore_risk: Whether cash or risk limits should be enforced or ignored for
                the order.
            tag:
        """

        order = LimitOrder(
            agent_id=self.id,
            time_placed=self.current_time,
            symbol=symbol,
            quantity=quantity,
            side=side,
            limit_price=limit_price,
            is_hidden=is_hidden,
            is_price_to_comply=is_price_to_comply,
            insert_by_id=insert_by_id,
            is_post_only=is_post_only,
            order_id=order_id,
            tag=tag,
        )

        if quantity != 0:
            # Test if this order can be permitted given our at-risk limits.
            new_position = self.position.copy()

            order_rate = tick_to_rate(order.limit_price)
            qty = order.quantity if order.side.is_bid() else -1 * order.quantity

            # Open margin is checked in place_limit_order
                
            return order

        else:
            warnings.warn(f"TradingAgent ignored limit order of quantity zero: {order}")

    def place_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        limit_price: int,
        order_id: Optional[int] = None,
        is_hidden: bool = False,
        is_price_to_comply: bool = False,
        insert_by_id: bool = False,
        is_post_only: bool = False,
        tag: Any = None,
    ) -> None:
        """
        Used by any Trading Agent subclass to place a limit order.

        Arguments:
            symbol: A valid symbol.
            quantity: Positive share quantity.
            side: Side.BID or Side.ASK.
            limit_price: Price in cents.
            order_id: An optional order id (otherwise global autoincrement is used).
            is_hidden:
            is_price_to_comply:
            insert_by_id:
            is_post_only:
            ignore_risk: Whether cash or risk limits should be enforced or ignored for
                the order.
            tag:
        """

        order = self.create_limit_order(
            symbol,
            quantity,
            side,
            limit_price,
            order_id,
            is_hidden,
            is_price_to_comply,
            insert_by_id,
            is_post_only,
            tag,
        )

        if order is not None:
            if not self.is_healthy():
                self.logEvent("FAILED_LIMIT_ORDER", self.last_R1, 
                              deepcopy_event=False)
            else:
                self.orders[order.order_id] = deepcopy(order)
                self.send_message(self.exchange_id, LimitOrderMsg(order))

            if self.log_orders:
                self.logEvent("ORDER_SUBMITTED", 
                              f"{order.side} {order.quantity} {order.symbol} @ {order.limit_price}", 
                              deepcopy_event=False)

    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        order_id: Optional[int] = None,
        ignore_risk: bool = True,
        tag: Any = None,
    ) -> None:
        """
        Used by any Trading Agent subclass to place a market order.

        The market order is created as multiple limit orders crossing the spread
        walking the book until all the quantities are matched.

        Arguments:
            symbol: Name of the stock traded.
            quantity: Order quantity.
            side: Side.BID or Side.ASK.
            order_id: Order ID for market replay.
            ignore_risk: Whether cash or risk limits should be enforced or ignored for
                the order.
            tag:
        """

        order = MarketOrder(
            self.id, self.current_time, symbol, quantity, side, order_id, tag
        )
        if quantity != 0:
            if not self.is_healthy():
                self.logEvent("FAILED_MARKET_ORDER", self.last_R1, 
                              deepcopy_event=False)
            else:
                self.orders[order.order_id] = deepcopy(order)
                self.send_message(self.exchange_id, MarketOrderMsg(order))
            if self.log_orders:
                self.logEvent("ORDER_SUBMITTED", 
                              f"{order.side} {order.quantity} {order.symbol} @ MARKET PRICE", 
                              deepcopy_event=False)

        else:
            warnings.warn(
                "TradingAgent ignored market order of quantity zero: {}", order
            )

    def place_multiple_orders(
        self, orders: List[Union[LimitOrder, MarketOrder]]
    ) -> None:
        """
        Used by any Trading Agent subclass to place multiple orders at the same time.

        Arguments:
            orders: A list of Orders to place with the exchange as a single batch.
        """

        messages = []

        for order in orders:
            if isinstance(order, LimitOrder):
                messages.append(LimitOrderMsg(order))
            elif isinstance(order, MarketOrder):
                messages.append(MarketOrderMsg(order))
            else:
                raise Exception("Expected LimitOrder or MarketOrder")

            # Copy the intended order for logging, so any changes made to it elsewhere
            # don't retroactively alter our "as placed" log of the order.  Eventually
            # it might be nice to make the whole history of the order into transaction
            # objects inside the order (we're halfway there) so there CAN be just a single
            # object per order, that never alters its original state, and eliminate all
            # these copies.
            self.orders[order.order_id] = deepcopy(order)

            if self.log_orders:
                self.logEvent("ORDER_SUBMITTED", 
                              f"{order.side} {order.quantity} {order.symbol} @ {order.limit_price}", 
                              deepcopy_event=False)

        if len(messages) > 0:
            self.send_message_batch(self.exchange_id, messages)

    def cancel_order(
        self, order: LimitOrder, tag: Optional[str] = None, metadata: dict = {}
    ) -> None:
        """
        Used by derived classes of TradingAgent to cancel a limit order.

        The order must currently appear in the agent's open orders list.

        Arguments:
            order: The limit order to cancel.
            tag:
            metadata:
        """

        if isinstance(order, LimitOrder):
            self.send_message(self.exchange_id, CancelOrderMsg(order, tag, metadata))
            if self.log_orders:
                self.logEvent("CANCEL_SUBMITTED", 
                              f"{order.side} {order.quantity} {order.symbol} @ {order.limit_price}", 
                              deepcopy_event=False)
        else:
            warnings.warn(f"Order {order} of type, {type(order)} cannot be cancelled")

    def cancel_all_orders(self):
        """
        Cancels all current limit orders held by this agent.
        """

        for order in self.orders.values():
            if isinstance(order, LimitOrder):
                self.cancel_order(order)

    def partial_cancel_order(
        self,
        order: LimitOrder,
        quantity: int,
        tag: Optional[str] = None,
        metadata: dict = {},
    ) -> None:
        """
        Used by any Trading Agent subclass to modify any existing limit order.

        The order must currently appear in the agent's open orders list.
        Arguments:
            order: The limit order to partially cancel.
            quantity:
            tag:
            metadata:
        """

        self.send_message(
            self.exchange_id, PartialCancelOrderMsg(order, quantity, tag, metadata)
        )

        if self.log_orders:
            self.logEvent("CANCEL_PARTIAL_ORDER", 
                              f"{order.side} {order.quantity} {order.symbol} @ {order.limit_price}", 
                              deepcopy_event=False)

    def modify_order(self, order: LimitOrder, new_order: LimitOrder) -> None:
        """
        Used by any Trading Agent subclass to modify any existing limit order.

        The order must currently appear in the agent's open orders list.  Some
        additional tests might be useful here to ensure the old and new orders are
        the same in some way.

        Arguments:
            order: The existing limit order.
            new_order: The limit order to update the existing order with.
        """

        self.send_message(self.exchange_id, ModifyOrderMsg(order, new_order))

        if self.log_orders:
            self.logEvent("MODIFY_ORDER",
                              f"{order.side} {order.quantity} {order.symbol} @ {order.limit_price}", 
                              deepcopy_event=False)

    def replace_order(self, order: LimitOrder, new_order: LimitOrder) -> None:
        """
        Used by any Trading Agent subclass to replace any existing limit order.

        The order must currently appear in the agent's open orders list.  Some
        additional tests might be useful here to ensure the old and new orders are
        the same in some way.

        Arguments:
            order: The existing limit order.
            new_order: The new limit order to replace the existing order with.
        """

        self.send_message(self.exchange_id, ReplaceOrderMsg(self.id, order, new_order))

        if self.log_orders:
            self.logEvent("REPLACE_ORDER", 
                              f"{order.side} {order.quantity} {order.symbol} @ {order.limit_price}", 
                              deepcopy_event=False)

    def order_executed(self, order: Order) -> None:
        """
        Handles OrderExecuted messages from an exchange agent.

        Subclasses may wish to extend, but should still call parent method for basic
        portfolio/returns tracking.

        Arguments:
            order: The order that has been executed by the exchange.
        """

        logger.debug(f"Received notification of execution for: {order}")

        if self.log_orders:
            self.logEvent("ORDER_EXECUTED", 
                              f"{order.side} {order.quantity} {order.symbol} @ {order.fill_price}", 
                              deepcopy_event=False)

        # At the very least, we must update position at execution time.
        qty = order.quantity if order.side.is_bid() else -1 * order.quantity

        # self.logEvent("SWAP_INFO", f"{qty}, {tick_to_rate(order.fill_price)}")
        self.position["SIZE"], self.position["FIXRATE"], p_merge_pa = merge_swap(self.position["SIZE"], self.position["FIXRATE"], 
                                                                     qty, tick_to_rate(order.fill_price))
            
        self.position["COLLATERAL"] += p_merge_pa*self.rate_normalizer*self.n_payment
        # self.logEvent("INFO", f"{p_merge_pa}, {self.rate_normalizer}, {self.n_payment}" )

        self.position_updated()

        # If this original order is now fully executed, remove it from the open orders list.
        # Otherwise, decrement by the quantity filled just now.  It is _possible_ that due
        # to timing issues, it might not be in the order list (i.e. we issued a cancellation
        # but it was executed first, or something).
        if order.order_id in self.orders:
            o = self.orders[order.order_id]

            if order.quantity >= o.quantity:
                del self.orders[order.order_id]
            else:
                o.quantity -= order.quantity

        else:
            warnings.warn(f"Execution received for order not in orders list: {order}")

        logger.debug(f"After order execution, agent open orders: {self.orders}")

    # PENDLE
    def swap(self, current_time: NanosecondTime, floating_rate: float):
        """
        PENDLE: Used periodically swap with the exchange based on pos_size and pos_fixrate. Only called by the kernel.

        Arguments:
            current_time: The time that this agent was notified to do the swap
            floating_rate: The rate (typically from Rate Oracle) user need to pay in exchange for the pos_fixrate

        Returns:
            
        """
        self.current_time = current_time

        self.position["COLLATERAL"] += self.position["SIZE"]*(floating_rate - self.position["FIXRATE"]*self.rate_normalizer)
        self.logEvent("SWAP", 
                      f"{round(100*floating_rate/self.rate_normalizer, 4)} %", 
                      deepcopy_event=False)
        self.n_payment -= 1
        assert self.n_payment >= 0, self.n_payment
        
        self.position_updated()
        
        return
    # END PENDLE

    def order_accepted(self, order: LimitOrder) -> None:
        """
        Handles OrderAccepted messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been accepted from the exchange.
        """

        logger.debug(f"Received notification of acceptance for: {order}")

        if self.log_orders:
            self.logEvent("ORDER_ACCEPTED", 
                              f"{order.side} {order.quantity} {order.symbol} @ {order.limit_price}", 
                              deepcopy_event=False)

        # We may later wish to add a status to the open orders so an agent can tell whether
        # a given order has been accepted or not (instead of needing to override this method).

    def order_cancelled(self, order: LimitOrder) -> None:
        """
        Handles OrderCancelled messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been cancelled by the exchange.
        """

        logger.debug(f"Received notification of cancellation for: {order}")

        if self.log_orders:
            self.logEvent("ORDER_CANCELLED", 
                              f"{order.side} {order.quantity} {order.symbol} @ {order.limit_price}", 
                              deepcopy_event=False)

        # Remove the cancelled order from the open orders list.  We may of course wish to have
        # additional logic here later, so agents can easily "look for" cancelled orders.  Of
        # course they can just override this method.
        if order.order_id in self.orders:
            del self.orders[order.order_id]
        else:
            warnings.warn(
                f"Cancellation received for order not in orders list: {order}"
            )

    def order_partial_cancelled(self, order: LimitOrder) -> None:
        """
        Handles OrderCancelled messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been partially cancelled by the exchange.
        """

        logger.debug(f"Received notification of partial cancellation for: {order}")

        if self.log_orders:
            self.logEvent("PARTIAL_CANCELLED",  
                              f"{order.side} {order.quantity} {order.symbol} @ {order.limit_price}", 
                              deepcopy_event=False)

        # if orders still in the list of agent's order update agent's knowledge of
        # current state of the order
        if order.order_id in self.orders:
            self.orders[order.order_id] = order

        else:
            warnings.warn(
                f"partial cancellation received for order not in orders list: {order}"
            )

        logger.debug(
            f"After order partial cancellation, agent open orders: {self.orders}"
        )

    def order_modified(self, order: LimitOrder) -> None:
        """
        Handles OrderModified messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been modified at the exchange.
        """

        logger.debug(f"Received notification of modification for: {order}")

        if self.log_orders:
            self.logEvent("ORDER_MODIFIED", 
                              f"{order.side} {order.quantity} {order.symbol} @ {order.limit_price}"
                              )

        # if orders still in the list of agent's order update agent's knowledge of
        # current state of the order
        if order.order_id in self.orders:
            self.orders[order.order_id] = order

        else:
            warnings.warn("Execution received for order not in orders list: {order}")

        logger.debug(f"After order modification, agent open orders: {self.orders}")

    def order_replaced(self, old_order: LimitOrder, new_order: LimitOrder) -> None:
        """
        Handles OrderReplaced messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been modified at the exchange.
        """

        logger.debug(f"Received notification of replacement for: {old_order}")

        if self.log_orders:
            self.logEvent("ORDER_REPLACED", 
                              f"{old_order.side} {old_order.quantity} {old_order.symbol} @ {old_order.limit_price}", 
                              deepcopy_event=False)

        # if orders still in the list of agent's order update agent's knowledge of
        # current state of the order
        if old_order.order_id in self.orders:
            del self.orders[old_order.order_id]
        else:
            warnings.warn(
                f"Execution received for order not in orders list: {old_order}"
            )

        self.orders[new_order.order_id] = new_order

        logger.debug(f"After order replacement, agent open orders: {self.orders}")

    def market_closed(self) -> None:
        """
        Handles MarketClosedMsg messages from an exchange agent.

        Subclasses may wish to extend.
        """

        logger.debug("Received notification of market closure.")

        # Log this activity.
        self.logEvent("MKT_CLOSED")

        # Remember that this has happened.
        self.mkt_closed = True

    def query_last_trade(self, symbol: str, price: int) -> None:
        """
        Handles QueryLastTradeResponseMsg messages from an exchange agent.

        Arguments:
            symbol: The symbol that was queried.
            price: The price at which the last trade executed at.
        """

        self.last_trade[symbol] = price

        logger.debug(
            "Received last trade price of {} for {}.".format(
                self.last_trade[symbol], symbol
            )
        )

        if self.mkt_closed:
            # Note this as the final price of the day.
            self.daily_close_price[symbol] = self.last_trade[symbol]

            logger.debug(
                "Received daily close price of {} for {}.".format(
                    self.last_trade[symbol], symbol
                )
            )

    def query_twap(self, twap: float) -> None:
        """
        Handles QueryTWAPResponseMsg messages from an exchange agent.

        Arguments:
            twap: The last twap queried.
        """

        self.last_twap = twap

    def query_spread(
        self,
        symbol: str,
        price: int,
        bids: List[List[Tuple[int, int]]],
        asks: List[List[Tuple[int, int]]],
        book: str,
    ) -> None:
        """
        Handles QuerySpreadResponseMsg messages from an exchange agent.

        Arguments:
            symbol: The symbol that was queried.
            price:
            bids:
            asks:
            book:
        """

        # The spread message now also includes last price for free.
        self.query_last_trade(symbol, price)

        self.known_bids[symbol] = bids
        self.known_asks[symbol] = asks

        if bids:
            best_bid, best_bid_qty = (bids[0][0], bids[0][1])
        else:
            best_bid, best_bid_qty = ("No bids", 0)
            logger.warning(f"No bid at {fmt_ts(self.current_time)}")

        if asks:
            best_ask, best_ask_qty = (asks[0][0], asks[0][1])
        else:
            best_ask, best_ask_qty = ("No asks", 0)
            logger.warning(f"No ask at {fmt_ts(self.current_time)}")

        logger.debug(
            "Received spread of {} @ {} / {} @ {} for {}".format(
                best_bid_qty, best_bid, best_ask_qty, best_ask, symbol
            )
        )

        # self.logEvent("BID_DEPTH", bids)
        # self.logEvent("ASK_DEPTH", asks)
        self.logEvent(
            "IMBALANCE", [sum([x[1] for x in bids]), sum([x[1] for x in asks])]
        )

        self.book = book

    def handle_market_data(self, message: MarketDataMsg) -> None:
        """
        Handles Market Data messages for agents using subscription mechanism.

        Arguments:
            message: The market data message,
        """

        if isinstance(message, L2DataMsg):
            symbol = message.symbol
            self.known_asks[symbol] = message.asks
            self.known_bids[symbol] = message.bids
            self.last_trade[symbol] = message.last_transaction
            self.exchange_ts[symbol] = message.exchange_ts

    def query_order_stream(self, symbol: str, orders) -> None:
        """
        Handles QueryOrderStreamResponseMsg messages from an exchange agent.

        It is up to the requesting agent to do something with the data, which is a list
        of dictionaries keyed by order id. The list index is 0 for orders since the most
        recent trade, 1 for orders that led up to the most recent trade, and so on.
        Agents are not given index 0 (orders more recent than the last trade).

        Arguments:
            symbol: The symbol that was queried.
            orders:
        """

        self.stream_history[symbol] = orders

    def query_transacted_volume(
        self, symbol: str, bid_volume: int, ask_volume: int
    ) -> None:
        """
        Handles the QueryTransactedVolResponseMsg messages from the exchange agent.

        Arguments:
            symbol: The symbol that was queried.
            bid_vol: The volume that has transacted on the bid side for the queried period.
            ask_vol: The volume that has transacted on the ask side for the queried period.
        """

        self.transacted_volume[symbol] = (bid_volume, ask_volume)

    # Utility functions that perform calculations from available knowledge, but implement no
    # particular strategy.

    def get_known_bid_ask(self, symbol: str, best: bool = True):
        """
        Extract the current known bid and asks.

        This does NOT request new information.

        Arguments:
            symbol: The symbol to query.
            best:
        """

        if best:
            bid = self.known_bids[symbol][0][0] if self.known_bids[symbol] else None
            ask = self.known_asks[symbol][0][0] if self.known_asks[symbol] else None
            bid_vol = self.known_bids[symbol][0][1] if self.known_bids[symbol] else 0
            ask_vol = self.known_asks[symbol][0][1] if self.known_asks[symbol] else 0
            return bid, bid_vol, ask, ask_vol
        else:
            bids = self.known_bids[symbol] if self.known_bids[symbol] else None
            asks = self.known_asks[symbol] if self.known_asks[symbol] else None
            return bids, asks

    def get_known_liquidity(self, symbol: str, within: float = 0.00) -> Tuple[int, int]:
        """
        Extract the current bid and ask liquidity within a certain proportion of the
        inside bid and ask.  (i.e. within=0.01 means to report total BID shares
        within 1% of the best bid price, and total ASK shares within 1% of the best
        ask price)

        Arguments:
            symbol: The symbol to query.
            within:

        Returns:
            (bid_liquidity, ask_liquidity).  Note that this is from the order book
            perspective, not the agent perspective.  (The agent would be selling into
            the bid liquidity, etc.)
        """

        bid_liq = self.get_book_liquidity(self.known_bids[symbol], within)
        ask_liq = self.get_book_liquidity(self.known_asks[symbol], within)

        logger.debug("Bid/ask liq: {}, {}".format(bid_liq, ask_liq))
        logger.debug("Known bids: {}".format(self.known_bids[self.symbol]))
        logger.debug("Known asks: {}".format(self.known_asks[self.symbol]))

        return bid_liq, ask_liq

    def get_book_liquidity(self, book: Iterable[Tuple[int, int]], within: float) -> int:
        """
        Helper function for the above.  Checks one side of the known order book.

        Arguments:
            book:
            within:
        """
        liq = 0
        for i, (price, shares) in enumerate(book):
            if i == 0:
                best = price

            # Is this price within "within" proportion of the best price?
            if abs(best - price) <= int(round(best * within)):
                logger.debug(
                    "Within {} of {}: {} with {} shares".format(
                        within, best, price, shares
                    )
                )
                liq += shares

        return liq

    # PENDLE
    def mark_to_market(
        self, position: Optional[Mapping[str, int]]=None, market_tick: int = None, log: bool = False
    ) -> int:
        """
        Marks position to market (including cash) based on the market rate from rate_oracle.

        Arguments:
            position:
            log: log this event or not
        """
        # If no position is provided, self evaluating the current position
        if not position:  
            position = self.position 
            log = True  # Only log the real mark_to_market

        # If market tick is not provided, find market_tick from orderbook directly
        if not market_tick:
            market_tick = self.last_twap  # Should be like this in the future
            market_tick = self.kernel.book.last_twap  # Temp for now

        cash = position["COLLATERAL"]
            
        value = (tick_to_rate(market_tick)-position["FIXRATE"])*self.size_normalized(position)

        cash += value

        if log:
            self.logEvent(
                "MARK_TO_MARKET",
                "{} + {} {} @ ({} - {}) = {}".format(
                    position["COLLATERAL"], self.size_normalized(position), self.symbol, tick_to_rate(market_tick), position["FIXRATE"], cash
                ),
            )

            self.logEvent("MARKED_TO_MARKET", cash)

        return cash
    
    # end PENDLE

    def get_known_bid_ask_midpoint(
        self, symbol: str
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Get the known best bid, ask, and bid/ask midpoint from cached data. No volume.

        Arguments:
            symbol: The symbol to query.
        """

        bid = self.known_bids[symbol][0][0] if self.known_bids[symbol] else None
        ask = self.known_asks[symbol][0][0] if self.known_asks[symbol] else None

        midpoint = (
            int(round((bid + ask) / 2)) if bid is not None and ask is not None else None
        )

        return bid, ask, midpoint

    def get_average_transaction_price(self) -> float:
        """Calculates the average price paid (weighted by the order size)."""

        return round(
            sum(
                executed_order.quantity * executed_order.fill_price
                for executed_order in self.executed_orders
            )
            / sum(executed_order.quantity for executed_order in self.executed_orders),
            2,
        )

    # PENDLE
    def maintainance_margin(self, size: Optional[float] = None, size_thresh: List[float] = [20, 100], mm_fac: List[float] = [0.03, 0.04, 0.05]) -> float:
        if size == None:
            size = self.position["SIZE"]

        size = abs(size)*self.n_payment*self.rate_normalizer  # Convert to positive
        mm = 0

        for i in range(len(size_thresh)):
            if size >= size_thresh[i]:
                surplus_size = size_thresh[i] if i==0 else size_thresh[i]-size_thresh[i-1]
                mm += mm_fac[i]*surplus_size
            else:
                surplus_size = size if i==0 else size-size_thresh[i-1]
                mm += mm_fac[i]*surplus_size
                return mm
            
        surplus_size = size-size_thresh[-1]
        mm += mm_fac[len(size_thresh)]*surplus_size
        
        return mm
        
    def mRatio(self, position: Optional[Mapping[str, int]]=None):
        if not position:
            position = self.position

        MtM = self.mark_to_market(position, log=False) 
        if position["SIZE"] == 0 or self.n_payment == 0:
            return np.inf if MtM>=0 else -np.inf
        return MtM/self.maintainance_margin(position["SIZE"])

    def is_healthy(self):
        if self.position["SIZE"] == 0 or self.n_payment == 0:
            return True if self.position["COLLATERAL"]>=0 else False
        
        if self.position["SIZE"] > 0:
            return self.last_R1 <= self.kernel.book.last_twap
        else:
            return self.last_R1 >= self.kernel.book.last_twap
    
    def merge_swap(self, size: float, rate: float):
        """
        Merge the current position with another position

        Arguments:
            size: The size of new position
            rate: The rate of new position
        """
        self.position["SIZE"], self.position["FIXRATE"], p_merge_pa = merge_swap(self.position["SIZE"], self.position["FIXRATE"],
                                                                                 size, rate)
        self.position["COLLATERAL"] += p_merge_pa

        assert self.position["COLLATERAL"] >= 0

        return p_merge_pa
    
    def liquidated(self, d_col, d_size):
        """
        Called after each successful liquidation. Might not be fully liquidated.

        Arguments:
        d_col: Loss in Collateral
        d_size: Decrease in position size
        """
        # After liquidated, log position.
        self.logEvent("LIQUIDATED", f"Col -{d_col}")
        self.position_updated()

    def logN1(self, size: float):
        self.logEvent("N1", size)

    def R1(self, position: Optional[Mapping[str, int]]=None) -> int:
        """
        Return the tick where mRatio = 1.

        Arguments:
        position: The position to be evaluated.
        """
        # If no position is provided, self evaluating the current position
        if not position:  
            position = self.position 
            mm = self.last_mm
        else:
            mm = self.maintainance_margin(position["SIZE"])

        if position["SIZE"]*self.n_payment == 0:
            return np.inf

        sensitive_rate = (mm - position["COLLATERAL"])/self.size_normalized(position) + position["FIXRATE"]

        sensitive_tick = rate_to_tick(sensitive_rate)

        return sensitive_tick
    
    def R2(self, position: Optional[Mapping[str, int]]=None) -> int:
        """
        Return the tick where mRatio = 0, i.e. MtM = 0

        Arguments:
        position: The position to be evaluated.
        """
        # If no position is provided, self evaluating the current position
        if not position:  
            position = self.position 

        if position["SIZE"]*self.n_payment == 0:
            return np.inf

        sensitive_rate = -position["COLLATERAL"]/(self.size_normalized(position)) + position["FIXRATE"]

        sensitive_tick = rate_to_tick(sensitive_rate)

        return sensitive_tick
    
    def size_normalized(self, position: Optional[Mapping[str, int]]=None):
        # If no position is provided, self evaluating the current position
        if not position:  
            position = self.position 

        return position["SIZE"]*self.n_payment*self.rate_normalizer
    
    def position_updated(self):
        self.logEvent("POSITION_UPDATED", str(self.position))
        self.last_mm = self.maintainance_margin()
        
        self.last_R1 = self.R1()
        self.last_R2 = self.R2()
        self.logEvent("METRIC", [self.position['SIZE'], self.last_R1, self.last_R2])
        
    def get_wake_frequency(self) -> NanosecondTime:
        return self.random_state.randint(low=0, high=100)