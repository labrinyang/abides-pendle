import pytest
import logging
import numpy as np
import pandas as pd

from abides_core.utils import str_to_ns
from abides_markets.agents import PendleMarketMakerAgent, ExchangeAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_core.kernel import Kernel
from abides_markets.orders import Side

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FakeOrderBook:
    def __init__(self):
        self.orders = []

    def receive_order(self, order):
        self.orders.append(order)

class FakeExchangeAgent(ExchangeAgent):
    def __init__(self, *args, **kwargs):
        kwargs.pop('use_metric_tracker', None)
        super().__init__(*args, **kwargs)
        self.order_book = FakeOrderBook()

    def receive_order(self, current_time, sender_id, order):
        self.order_book.receive_order(order)

class PendleMarketMakerAgentTestHelper(PendleMarketMakerAgent):
    def __init__(self, exchange_agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exchange_agent = exchange_agent

    def place_multiple_orders(self, orders):
        for order in orders:
            self.exchange_agent.receive_order(self.current_time, self.id, order)

def test_pendle_market_maker_agent_specific_example():
    pov = 0.025
    orders_size = [{"time": 0.2, "size": 1000}, {"time": 0.5, "size": 2000}, {"time": 1.0, "size": 1500}]
    window_size, num_ticks, level_spacing, poisson_arrival = 10, 5, 0.5, False
    min_imbalance, cancel_limit_delay, wake_up_freq, r_bar = 0.9, 0, 1 * 60 * 60 * 1_000_000_000, 1000

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

    kernel = Kernel(agents=[exchange_agent, pendle_agent], swap_interval=str_to_ns("1h"))
    pendle_agent.kernel, exchange_agent.kernel = kernel, kernel

    mkt_open, mkt_close = str_to_ns("09:00:00"), str_to_ns("17:00:00")
    pendle_agent.mkt_open, pendle_agent.mkt_close, pendle_agent.current_time, pendle_agent.exchange_id = mkt_open, mkt_close, mkt_open, 0

    expected_orders_step1 = {
        'bids': [{'price': 995, 'quantity': 10_000}, {'price': 970, 'quantity': 9_000}],
        'asks': [{'price': 1005, 'quantity': 10_000}, {'price': 1030, 'quantity': 9_000}]
    }

    pendle_agent.wakeup(mkt_open)

    spread_response_msg = QuerySpreadResponseMsg(
        symbol="PEN",
        bids=[(990, 1)],
        asks=[(1010, 1)],
        mkt_closed=False,
        depth=1,
        last_trade=None
    )
    pendle_agent.receive_message(mkt_open, sender_id=0, message=spread_response_msg)

    actual_orders_step1 = {'bids': [], 'asks': []}
    for order in exchange_agent.order_book.orders:
        if order.side == Side.BID:
            actual_orders_step1['bids'].append({'price': order.limit_price, 'quantity': order.quantity})
        elif order.side == Side.ASK:
            actual_orders_step1['asks'].append({'price': order.limit_price, 'quantity': order.quantity})

    exchange_agent.order_book.orders = []

    assert len(actual_orders_step1['bids']) == len(expected_orders_step1['bids'])
    assert len(actual_orders_step1['asks']) == len(expected_orders_step1['asks'])
    for actual, expected in zip(actual_orders_step1['bids'], expected_orders_step1['bids']):
        assert actual['price'] == expected['price']
        assert actual['quantity'] == expected['quantity']
    for actual, expected in zip(actual_orders_step1['asks'], expected_orders_step1['asks']):
        assert actual['price'] == expected['price']
        assert actual['quantity'] == expected['quantity']

    logger.info("Step 1 orders generated correctly.")
