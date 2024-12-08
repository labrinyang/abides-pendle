import pytest
import logging
import numpy as np
import os
import pandas as pd

from abides_core import NanosecondTime
from abides_core.utils import str_to_ns, fmt_ts
from abides_markets.agents import PendleMarketMakerAgent, ExchangeAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.messages.marketdata import BookImbalanceDataMsg, MarketDataEventMsg
from abides_core.kernel import Kernel
from abides_markets.orders import Side  

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define a FakeOrderBook to log received orders
class FakeOrderBook:
    def __init__(self):
        self.orders = []

    def receive_order(self, order):
        self.orders.append(order)

# Define a FakeExchangeAgent to simulate a market and record orders
class FakeExchangeAgent(ExchangeAgent):
    def __init__(self, *args, **kwargs):
        # Remove unsupported parameters
        kwargs.pop('use_metric_tracker', None)
        super().__init__(*args, **kwargs)
        self.order_book = FakeOrderBook()

    def receive_order(self, current_time, sender_id, order):
        logger.debug(f"ExchangeAgent received order from Agent {sender_id}: {order}")
        self.order_book.receive_order(order)

# Define a PendleMarketMakerAgentTestHelper subclass for testing, overriding place_multiple_orders
class PendleMarketMakerAgentTestHelper(PendleMarketMakerAgent):
    def __init__(self, exchange_agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exchange_agent = exchange_agent

    def place_multiple_orders(self, orders):
        """
        Override the parent class method to send orders directly to the exchange agent's order book.
        """
        for order in orders:
            self.exchange_agent.receive_order(self.current_time, self.id, order)

# Define test cases
def test_pendle_market_maker_agent_specific_example():
    """
    Test whether PendleMarketMakerAgent generates orders correctly under specific parameters and market conditions.
    """
    # Define test parameters
    pov = 0.025
    orders_size = [
        {"time": 0.2, "size": 1000},
        {"time": 0.5, "size": 2000},
        {"time": 1.0, "size": 1500}
    ]
    window_size = 10
    num_ticks = 5
    level_spacing = 0.5
    poisson_arrival = False
    min_imbalance = 0.9
    cancel_limit_delay = 0  
    wake_up_freq = 1 * 60 * 60 * 1_000_000_000  
    r_bar = 1000

    exchange_agent = FakeExchangeAgent(
        id=0,
        mkt_open=str_to_ns("09:00:00"),
        mkt_close=str_to_ns("17:00:00"),
        symbols=["PEN"],
        name="TestExchange",
        type="ExchangeAgent",
        random_state=np.random.RandomState(seed=42),
        log_orders=False
    )


    pendle_agent = PendleMarketMakerAgentTestHelper(
        exchange_agent=exchange_agent,
        id=1,
        symbol="PEN",
        pov=pov,
        orders_size=orders_size,
        window_size=window_size,
        num_ticks=num_ticks,
        level_spacing=level_spacing,
        poisson_arrival=poisson_arrival,
        min_imbalance=min_imbalance,
        cancel_limit_delay=cancel_limit_delay,
        wake_up_freq=wake_up_freq,
        r_bar=r_bar,
        random_state=np.random.RandomState(seed=43),
        log_orders=False
    )

    # Create Kernel and add agents
    kernel = Kernel(
        agents=[exchange_agent, pendle_agent],
        swap_interval=str_to_ns("1h")
    )
    pendle_agent.kernel = kernel
    exchange_agent.kernel = kernel

    # Define market open and close times
    mkt_open = str_to_ns("09:00:00")
    mkt_close = str_to_ns("17:00:00")
    pendle_agent.mkt_open = mkt_open
    pendle_agent.mkt_close = mkt_close
    pendle_agent.current_time = mkt_open
    pendle_agent.exchange_id = 0

    # Initialize tick_size
    initial_spread_value = 50  # Assume initial spread is 50
    tick_size = int(np.ceil(initial_spread_value * level_spacing))  # tick_size = 25

    # Set initial mid price
    last_mid = r_bar  # 1000

    # Expected orders at step 1
    expected_orders_step1 = {
        'bids': [
            {'price': 995, 'quantity': 10_000},
            {'price': 970, 'quantity': 9_000},
            {'price': 945, 'quantity': 8_000},
            {'price': 920, 'quantity': 7_000},
            {'price': 895, 'quantity': 6_000},
        ],
        'asks': [
            {'price': 1005, 'quantity': 10_000},
            {'price': 1030, 'quantity': 9_000},
            {'price': 1055, 'quantity': 8_000},
            {'price': 1080, 'quantity': 7_000},
            {'price': 1105, 'quantity': 6_000},
        ]
    }

    # Step 1: First wakeup (09:00)
    pendle_agent.wakeup(mkt_open)

    # Simulate receiving spread information
    bid = 990  # 990 cents or $9.90
    ask = 1010  # 1010 cents or $10.10
    spread_response_msg = QuerySpreadResponseMsg(
        symbol="PEN",
        bids=[(bid, 1)],  # Keep in cents
        asks=[(ask, 1)],
        mkt_closed=False,
        depth=1,
        last_trade=None
    )
    pendle_agent.receive_message(mkt_open, sender_id=0, message=spread_response_msg)

    # Collect actual generated orders
    actual_orders_step1 = {
        'bids': [],
        'asks': []
    }
    for order in exchange_agent.order_book.orders:
        if order.side == Side.BID:
            actual_orders_step1['bids'].append({'price': order.limit_price, 'quantity': order.quantity})
        elif order.side == Side.ASK:
            actual_orders_step1['asks'].append({'price': order.limit_price, 'quantity': order.quantity})

    # Clear order book
    exchange_agent.order_book.orders = []

    # Verify step 1 orders
    assert len(actual_orders_step1['bids']) == len(expected_orders_step1['bids']), "Mismatch in bid orders (Step 1)"
    assert len(actual_orders_step1['asks']) == len(expected_orders_step1['asks']), "Mismatch in ask orders (Step 1)"

    for actual, expected in zip(actual_orders_step1['bids'], expected_orders_step1['bids']):
        assert actual['price'] == expected['price'], f"Mismatch in bid price (Step 1): {actual['price']} != {expected['price']}"
        assert actual['quantity'] == expected['quantity'], f"Mismatch in bid quantity (Step 1): {actual['quantity']} != {expected['quantity']}"

    for actual, expected in zip(actual_orders_step1['asks'], expected_orders_step1['asks']):
        assert actual['price'] == expected['price'], f"Mismatch in ask price (Step 1): {actual['price']} != {expected['price']}"
        assert actual['quantity'] == expected['quantity'], f"Mismatch in ask quantity (Step 1): {actual['quantity']} != {expected['quantity']}"

    logger.info("Step 1 orders generated correctly.")
