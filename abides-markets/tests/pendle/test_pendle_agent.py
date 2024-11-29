import logging
import numpy as np
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns, fmt_ts
from abides_markets.agents import PendleSeedingAgent, ExchangeAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from abides_core.kernel import Kernel
from typing import Any, Dict, List, Optional, Tuple
from abides_core.message import Message, MessageBatch, WakeupMsg, SwapMsg, UpdateRateMsg
from abides_markets.messages.order import LimitOrderMsg, MarketOrderMsg
from abides_markets.order_book import OrderBook
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class FakeRateOracle:
    def __init__(self, fixed_rate=0.05):
        self.fixed_rate = fixed_rate
        logger.debug(f"FakeRateOracle 初始化完成，固定利率: {self.fixed_rate}")

    def get_floating_rate(self, current_time: NanosecondTime) -> float:
        logger.debug(f"获取浮动利率，当前时间: {current_time}")
        return self.fixed_rate

def test_pendle_seeding_agent_with_exchange():
    """
    测试 PendleSeedingAgent 确保在订单簿为空时正确地注入市场流动性。
    """
    logger.debug("开始测试 PendleSeedingAgent 与 ExchangeAgent 的交互")

    # 初始化 ExchangeAgent（通常为 id=0）
    logger.debug("初始化 ExchangeAgent")
    exchange_agent = ExchangeAgent(
        id=0,
        mkt_open=0,
        mkt_close=365 * str_to_ns("1d"),  # 1年市场关闭时间
        symbols=["PEN"],
        name="TestExchange",
        type="ExchangeAgent",
        random_state=np.random.RandomState(seed=42),
        log_orders=False,
        use_metric_tracker=True  # 设置为 True
    )

    # 初始化 PendleSeedingAgent（id=1）
    agent_id = 1
    symbol = "PEN"
    size = 100
    min_bid = 1
    max_bid = 10
    min_ask = 11
    max_ask = 20

    logger.debug(f"初始化 PendleSeedingAgent，ID={agent_id}，符号='{symbol}'")
    logger.debug(f"订单大小: {size}, Bid 价格范围: {min_bid}-{max_bid}, Ask 价格范围: {min_ask}-{max_ask}")

    # 创建一个随机状态以确保测试的可重复性
    random_state = np.random.RandomState(seed=42)
    logger.debug("创建随机状态以确保测试的可重复性")

    # 实例化 PendleSeedingAgent
    seeding_agent = PendleSeedingAgent(
        id=agent_id,
        symbol=symbol,
        random_state=random_state,
        size=size,
        min_bid=min_bid,
        max_bid=max_bid,
        min_ask=min_ask,
        max_ask=max_ask
    )
    logger.debug("PendleSeedingAgent 实例化完成")

    # 设置 wakeup_time
    wakeup_time = str_to_ns("0.1s")  # 0.1秒转换为纳秒
    logger.debug(f"设置 wakeup_time 为 {wakeup_time}")
    
    # 创建并配置 Kernel，传递 ExchangeAgent 和 PendleSeedingAgent
    logger.debug("创建并配置 Kernel，包含 ExchangeAgent 和 PendleSeedingAgent")
    kernel = Kernel(
        agents=[exchange_agent, seeding_agent],
        swap_interval=str_to_ns("8h"),
        start_time=str_to_ns("00:00:00"),
        stop_time=str_to_ns("0.5s")  # 设置 stop_time 为 0.5 秒
    )

    # 设置 rate_oracle
    fake_rate_oracle = FakeRateOracle(fixed_rate=0.05)
    kernel.rate_oracle = fake_rate_oracle
    logger.debug("将 FakeRateOracle 分配给 Kernel")

    # 将 Kernel 分配给代理
    logger.debug("将 Kernel 分配给代理")
    seeding_agent.kernel = kernel
    exchange_agent.kernel = kernel

    # 设置 PendleSeedingAgent 的属性
    logger.debug("设置 PendleSeedingAgent 的属性")
    seeding_agent.mkt_open = 1
    seeding_agent.mkt_close = str_to_ns("1s")  # 任意市场关闭时间为1秒
    seeding_agent.current_time = 0
    seeding_agent.exchange_id = 0  # 指向 ExchangeAgent 的 id

    # 模拟已知的买卖盘数据为空（订单簿为空）
    logger.debug("通过设置 known_bids 和 known_asks 为空列表，模拟订单簿为空")
    seeding_agent.known_bids = {symbol: []}
    seeding_agent.known_asks = {symbol: []}

    # 调度 PendleSeedingAgent 的 wakeup 调用
    logger.debug(f"调度 PendleSeedingAgent 在时间 {wakeup_time} 唤醒")
    kernel.set_wakeup(seeding_agent.id, wakeup_time)  # 使用 set_wakeup 方法

    # 运行 Kernel
    logger.debug(f"运行 Kernel 直到 stop_time {str_to_ns('0.5s')}")
    kernel.run()

    # 验证 ExchangeAgent 的订单簿中是否包含 PendleSeedingAgent 下达的订单
    logger.debug("验证 ExchangeAgent 的订单簿中是否包含 PendleSeedingAgent 下达的订单")

    expected_bid_prices = list(range(min_bid, max_bid + 1))
    expected_ask_prices = list(range(min_ask, max_ask + 1))

    # 从 ExchangeAgent 的订单簿中提取 bid 和 ask 订单
    placed_bids = [
        order for price_level in exchange_agent.order_books[symbol].bids
        for order, _ in price_level.visible_orders
        if order.side == Side.BID and order.symbol == symbol
    ]
    placed_asks = [
        order for price_level in exchange_agent.order_books[symbol].asks
        for order, _ in price_level.visible_orders
        if order.side == Side.ASK and order.symbol == symbol
    ]

    logger.debug(f"预期的 bid 价格: {expected_bid_prices}")
    logger.debug(f"预期的 ask 价格: {expected_ask_prices}")
    logger.debug(f"实际放置的 bid 订单价格: {[order.limit_price for order in placed_bids]}")
    logger.debug(f"实际放置的 ask 订单价格: {[order.limit_price for order in placed_asks]}")

    # 检查放置的 bid 和 ask 订单数量是否正确
    assert len(placed_bids) == len(expected_bid_prices), \
        f"预期有 {len(expected_bid_prices)} 个 bid 订单，但实际有 {len(placed_bids)} 个"
    assert len(placed_asks) == len(expected_ask_prices), \
        f"预期有 {len(expected_ask_prices)} 个 ask 订单，但实际有 {len(placed_asks)} 个"

    # 验证每个 bid 订单是否正确
    for price in expected_bid_prices:
        matching_orders = [
            order for order in placed_bids 
            if order.limit_price == price and order.quantity == size
        ]
        assert len(matching_orders) == 1, f"预期在价格 {price} 有一个 bid 订单，但实际有 {len(matching_orders)} 个"

    # 验证每个 ask 订单是否正确
    for price in expected_ask_prices:
        matching_orders = [
            order for order in placed_asks 
            if order.limit_price == price and order.quantity == size
        ]
        assert len(matching_orders) == 1, f"预期在价格 {price} 有一个 ask 订单，但实际有 {len(matching_orders)} 个"

    # 额外的断言：验证交易量
    bid_volume, ask_volume = exchange_agent.order_books[symbol].get_transacted_volume(
        lookback_period=str_to_ns("10min")
    )
    assert bid_volume == 0, f"预期买单交易量为 0，但实际为 {bid_volume}"
    assert ask_volume == 0, f"预期卖单交易量为 0，但实际为 {ask_volume}"

    logger.info("PendleSeedingAgent 与 ExchangeAgent 的测试成功通过。")