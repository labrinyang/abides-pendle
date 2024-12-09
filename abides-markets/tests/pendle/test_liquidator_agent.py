import pytest
import logging
import numpy as np
from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns
from abides_markets.agents import TradingAgent, LiquidatorAgent, ExchangeAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.messages.market import MarketHoursMsg
from abides_markets.orders import Side
from abides_core.kernel import Kernel
from abides_core.message import WakeupMsg

logger = logging.getLogger(__name__)
FakeKernel = Kernel

class FakeOrderBook:
    def __init__(self):
        self.last_twap = -3029.555281

class FakeRateOracle:
    def __init__(self):
        pass

    def get_floating_rate(self, current_time: NanosecondTime) -> float:
        return 0


def test_liquidator_calculate_into_bid_wall():
    logger.debug("Starting test_liquidator_calculate_into_bid_wall")

    mkt_open = 0
    mkt_close = 365 * str_to_ns("1d")
    exchange_agent = ExchangeAgent(
        id=0,
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        symbols=["PEN"],
        name="TestExchange",
        type="ExchangeAgent",
        random_state=np.random.RandomState(seed=42),
        log_orders=False,
        use_metric_tracker=False
    )

    exchange_agent.order_books["PEN"] = FakeOrderBook()

    liquidator = LiquidatorAgent(id=1, name="LiquidatorAgent")
    unhealthy_agent = TradingAgent(id=2, name="UnhealthyAgent")

    swap_interval = str_to_ns("8h")
    kernel = Kernel(
        agents=[exchange_agent, liquidator, unhealthy_agent],
        swap_interval=swap_interval,
        custom_properties={"rate_oracle": FakeRateOracle()}
    )

    liquidator.kernel = kernel
    unhealthy_agent.kernel = kernel
    exchange_agent.kernel = kernel

    liquidator.kernel_starting(start_time=0)
    unhealthy_agent.kernel_starting(start_time=0)

    market_hours_msg = MarketHoursMsg(
        mkt_open=exchange_agent.mkt_open,
        mkt_close=exchange_agent.mkt_close
    )
    liquidator.receive_message(
        current_time=0,
        sender_id=exchange_agent.id,
        message=market_hours_msg
    )
    unhealthy_agent.receive_message(
        current_time=0,
        sender_id=exchange_agent.id,
        message=market_hours_msg
    )

    liquidator.exchange_id = exchange_agent.id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(-3027.555281, 5000), (-3032.555281, 5000)]}
    liquidator.known_asks = {"PEN": [(-3027.555281, -5000), (-3026.555281, -5000)]}
    liquidator.mkt_closed = False
    liquidator.watch_list = [unhealthy_agent.id]

    unhealthy_agent.position = {"COLLATERAL": 50, "SIZE": 100, "FIXRATE": 0.1095}
    unhealthy_agent.exchange_id = exchange_agent.id

    unhealthy_agent.position_updated()
    liquidator.position_updated()

    health_status = unhealthy_agent.is_healthy()
    logger.debug(f"position: {unhealthy_agent.position}, health_status: {health_status}")
    assert not health_status

    current_time = 2
    liquidator.current_time = current_time

    message = QuerySpreadResponseMsg(
        symbol=liquidator.symbol,
        bids=liquidator.known_bids[liquidator.symbol],
        asks=liquidator.known_asks[liquidator.symbol],
        mkt_closed=False,
        depth=2,
        last_trade=None,
    )

    liquidator.receive_message(current_time, sender_id=exchange_agent.id, message=message)
    liquidator.check_liquidate(unhealthy_agent)
    assert round(unhealthy_agent.position["COLLATERAL"], 2) == 3.15, f"Expected 3.53, got {unhealthy_agent.position['COLLATERAL']}"
    logger.debug(f"After liquidation, unhealthy_agent.position['COLLATERAL']: {unhealthy_agent.position['COLLATERAL']}")

    assert unhealthy_agent.position["SIZE"] == 0
    logger.debug(f"After liquidation, unhealthy_agent.position['SIZE']: {unhealthy_agent.position['SIZE']}")

    assert round(liquidator.position["COLLATERAL"], 2) == 100046.85
    logger.debug(f"After liquidation, liquidator.position['COLLATERAL']: {liquidator.position['COLLATERAL']}")

    assert round(liquidator.position["FIXRATE"], 2) == 0.11
    logger.debug(f"After liquidation, liquidator.position['FIXRATE']: {liquidator.position['FIXRATE']}")

def test_liquidator_calculate_into_ask_wall():
    logger.debug("Starting test_liquidator_calculate_into_ask_wall")

    mkt_open = 0
    mkt_close = 365 * str_to_ns("1d")
    exchange_agent = ExchangeAgent(
        id=0,
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        symbols=["PEN"],
        name="TestExchange",
        type="ExchangeAgent",
        random_state=np.random.RandomState(seed=42),
        log_orders=False,
        use_metric_tracker=False
    )

    exchange_agent.order_books["PEN"] = FakeOrderBook()

    liquidator = LiquidatorAgent(id=1, name="LiquidatorAgent")
    unhealthy_agent = TradingAgent(id=2, name="UnhealthyAgent")

    swap_interval = str_to_ns("8h")
    kernel = Kernel(
        agents=[exchange_agent, liquidator, unhealthy_agent],
        swap_interval=swap_interval,
        custom_properties={"rate_oracle": FakeRateOracle()}
    )

    liquidator.kernel = kernel
    unhealthy_agent.kernel = kernel
    exchange_agent.kernel = kernel

    liquidator.kernel_starting(start_time=0)
    unhealthy_agent.kernel_starting(start_time=0)

    market_hours_msg = MarketHoursMsg(
        mkt_open=exchange_agent.mkt_open,
        mkt_close=exchange_agent.mkt_close
    )
    liquidator.receive_message(
        current_time=0,
        sender_id=exchange_agent.id,
        message=market_hours_msg
    )
    unhealthy_agent.receive_message(
        current_time=0,
        sender_id=exchange_agent.id,
        message=market_hours_msg
    )

    liquidator.exchange_id = exchange_agent.id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(-3030.555281, 5000), (-3032.555281, 5000)]}
    liquidator.known_asks = {"PEN": [(-3030.555281, 5000), (-3026.555281, 5000)]}
    liquidator.mkt_closed = False
    liquidator.watch_list = [unhealthy_agent.id]

    unhealthy_agent.position = {"COLLATERAL": 50, "SIZE": -100, "FIXRATE": -0.816}
    unhealthy_agent.exchange_id = exchange_agent.id

    unhealthy_agent.position_updated()
    liquidator.position_updated()

    health_status = unhealthy_agent.is_healthy()
    logger.debug(f"position: {unhealthy_agent.position}, health_status: {health_status}")
    assert not health_status

    current_time = 2
    liquidator.current_time = current_time

    message = QuerySpreadResponseMsg(
        symbol=liquidator.symbol,
        bids=liquidator.known_bids[liquidator.symbol],
        asks=liquidator.known_asks[liquidator.symbol],
        mkt_closed=False,
        depth=2,
        last_trade=None,
    )

    liquidator.receive_message(current_time, sender_id=exchange_agent.id, message=message)
    liquidator.check_liquidate(unhealthy_agent)
    assert round(unhealthy_agent.position["COLLATERAL"], 2) == 3.39, f"Expected 3.77, got {unhealthy_agent.position['COLLATERAL']}"
    logger.debug(f"After liquidation, unhealthy_agent.position['COLLATERAL']: {unhealthy_agent.position['COLLATERAL']}")

    assert unhealthy_agent.position["SIZE"] == 0
    logger.debug(f"After liquidation, unhealthy_agent.position['SIZE']: {unhealthy_agent.position['SIZE']}")

    assert round(liquidator.position["COLLATERAL"], 2) == 100046.61
    logger.debug(f"After liquidation, liquidator.position['COLLATERAL']: {liquidator.position['COLLATERAL']}")

    assert round(liquidator.position["FIXRATE"], 2) == -0.82
    logger.debug(f"After liquidation, liquidator.position['FIXRATE']: {liquidator.position['FIXRATE']}")
