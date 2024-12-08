# test_market_maker_agent.py

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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define a FakeOrderBook to record received orders
class FakeOrderBook:
    def __init__(self):
        self.orders = []

    def receive_order(self, order):
        self.orders.append(order)

# Define a FakeExchangeAgent to simulate the market and record orders
class FakeExchangeAgent(ExchangeAgent):
    def __init__(self, *args, **kwargs):
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
        Override parent method to send orders directly to exchange_agent's order_book.
        """
        for order in orders:
            self.exchange_agent.receive_order(self.current_time, self.id, order)

# Define test case
def test_pendle_market_maker_agent_specific_example():
    """
    Test if PendleMarketMakerAgent generates orders correctly under specific parameters and market conditions.
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
    cancel_limit_delay = 0  # Nanoseconds
    wake_up_freq = 1 * 60 * 60 * 1_000_000_000  # 1 hour in nanoseconds
    r_bar = 1000

    # Create ExchangeAgent
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

    # Create PendleMarketMakerAgentTestHelper
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

    # Define market open time
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

    # Define expected orders (in cents)
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

    # Define expected step2 orders (same as step1 because time_now=0.125 <= 0.2)
    expected_orders_step2 = expected_orders_step1.copy()

    # Define expected orders when handling imbalance (quantity doubled)
    expected_orders_step3 = {
        'bids': [
            {'price': 995, 'quantity': 20_000},
            {'price': 970, 'quantity': 18_000},
            {'price': 945, 'quantity': 16_000},
            {'price': 920, 'quantity': 14_000},
            {'price': 895, 'quantity': 12_000},
        ],
        'asks': [
            {'price': 1005, 'quantity': 20_000},
            {'price': 1030, 'quantity': 18_000},
            {'price': 1055, 'quantity': 16_000},
            {'price': 1080, 'quantity': 14_000},
            {'price': 1105, 'quantity': 12_000},
        ]
    }

    # Simulate time steps and events

    # Step 1: First wakeup (09:00)
    pendle_agent.wakeup(mkt_open)
    
    bid = 990  # tick
    ask = 1010  
    spread_response_msg = QuerySpreadResponseMsg(
        symbol="PEN",
        bids=[(bid, 1)],
        asks=[(ask, 1)],
        mkt_closed=False,
        depth=1,
        last_trade=None
    )
    pendle_agent.receive_message(mkt_open, sender_id=0, message=spread_response_msg)

    # Get actual generated orders
    actual_orders_step1 = {
        'bids': [],
        'asks': []
    }
    for order in exchange_agent.order_book.orders:
        if order.side == Side.BID:
            actual_orders_step1['bids'].append({'price': order.limit_price, 'quantity': order.quantity})
        elif order.side == Side.ASK:
            actual_orders_step1['asks'].append({'price': order.limit_price, 'quantity': order.quantity})

    # Clear order records
    exchange_agent.order_book.orders = []

    # Verify step1 orders
    assert len(actual_orders_step1['bids']) == len(expected_orders_step1['bids']), "Number of bids does not match (Step1)"
    assert len(actual_orders_step1['asks']) == len(expected_orders_step1['asks']), "Number of asks does not match (Step1)"

    for actual, expected in zip(actual_orders_step1['bids'], expected_orders_step1['bids']):
        assert actual['price'] == expected['price'], f"Bid price does not match (Step1): {actual['price']} != {expected['price']}"
        assert actual['quantity'] == expected['quantity'], f"Bid quantity does not match (Step1): {actual['quantity']} != {expected['quantity']}"

    for actual, expected in zip(actual_orders_step1['asks'], expected_orders_step1['asks']):
        assert actual['price'] == expected['price'], f"Ask price does not match (Step1): {actual['price']} != {expected['price']}"
        assert actual['quantity'] == expected['quantity'], f"Ask quantity does not match (Step1): {actual['quantity']} != {expected['quantity']}"

    logger.info("Orders generated correctly in Step1.")

    # Step 2: Next wakeup (10:00)
    next_wakeup_time = str_to_ns("10:00:00")
    pendle_agent.wakeup(next_wakeup_time)

    # Simulate receiving new spread information
    new_bid = 980 # tick
    new_ask = 1020  
    new_spread_response_msg = QuerySpreadResponseMsg(
        symbol="PEN",
        bids=[(new_bid, 1)],
        asks=[(new_ask, 1)],
        mkt_closed=False,
        depth=1,
        last_trade=None
    )
    pendle_agent.receive_message(next_wakeup_time, sender_id=0, message=new_spread_response_msg)

    # Get actual generated orders
    actual_orders_step2 = {
        'bids': [],
        'asks': []
    }
    for order in exchange_agent.order_book.orders:
        if order.side == Side.BID:
            actual_orders_step2['bids'].append({'price': order.limit_price, 'quantity': order.quantity})
        elif order.side == Side.ASK:
            actual_orders_step2['asks'].append({'price': order.limit_price, 'quantity': order.quantity})

    # Clear order records
    exchange_agent.order_book.orders = []

    # Verify step2 orders
    assert len(actual_orders_step2['bids']) == len(expected_orders_step2['bids']), "Number of bids does not match (Step2)"
    assert len(actual_orders_step2['asks']) == len(expected_orders_step2['asks']), "Number of asks does not match (Step2)"

    for actual, expected in zip(actual_orders_step2['bids'], expected_orders_step2['bids']):
        assert actual['price'] == expected['price'], f"Bid price does not match (Step2): {actual['price']} != {expected['price']}"
        assert actual['quantity'] == expected['quantity'], f"Bid quantity does not match (Step2): {actual['quantity']} != {expected['quantity']}"

    for actual, expected in zip(actual_orders_step2['asks'], expected_orders_step2['asks']):
        assert actual['price'] == expected['price'], f"Ask price does not match (Step2): {actual['price']} != {expected['price']}"
        assert actual['quantity'] == expected['quantity'], f"Ask quantity does not match (Step2): {actual['quantity']} != {expected['quantity']}"

    logger.info("Orders generated correctly in Step2.")

    # Step 3: Handle market imbalance (11:00)
    imbalance_time = str_to_ns("11:00:00")
    imbalance = 0.95  
    imbalance_side = Side.BID

    imbalance_msg = BookImbalanceDataMsg(
        symbol="PEN",
        last_transaction=int(imbalance_time),
        exchange_ts=pd.Timestamp(imbalance_time / 1e9, unit='s'),
        stage=MarketDataEventMsg.Stage.START,
        imbalance=imbalance,
        side=imbalance_side
    )
    pendle_agent.receive_message(imbalance_time, sender_id=0, message=imbalance_msg)

    # Get actual generated orders
    actual_orders_step3 = {
        'bids': [],
        'asks': []
    }
    for order in exchange_agent.order_book.orders:
        if order.side == Side.BID:
            actual_orders_step3['bids'].append({'price': order.limit_price, 'quantity': order.quantity})
        elif order.side == Side.ASK:
            actual_orders_step3['asks'].append({'price': order.limit_price, 'quantity': order.quantity})

    # Clear order records
    exchange_agent.order_book.orders = []

    # Verify step3 orders
    assert len(actual_orders_step3['bids']) == len(expected_orders_step3['bids']), "Number of bids does not match (Step3)"
    assert len(actual_orders_step3['asks']) == len(expected_orders_step3['asks']), "Number of asks does not match (Step3)"

    for actual, expected in zip(actual_orders_step3['bids'], expected_orders_step3['bids']):
        assert actual['price'] == expected['price'], f"Bid price does not match (Step3): {actual['price']} != {expected['price']}"
        assert actual['quantity'] == expected['quantity'], f"Bid quantity does not match (Step3): {actual['quantity']} != {expected['quantity']}"

    for actual, expected in zip(actual_orders_step3['asks'], expected_orders_step3['asks']):
        assert actual['price'] == expected['price'], f"Ask price does not match (Step3): {actual['price']} != {expected['price']}"
        assert actual['quantity'] == expected['quantity'], f"Ask quantity does not match (Step3): {actual['quantity']} != {expected['quantity']}"

    logger.info("Orders generated correctly in Step3.")

    # End test
    logger.info("PendleMarketMakerAgent test passed, all orders generated as expected.")
