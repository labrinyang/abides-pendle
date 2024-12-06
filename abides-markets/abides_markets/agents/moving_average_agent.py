import logging
from typing import Optional, List
import numpy as np
from abides_core import Message, NanosecondTime
from abides_core.message import QueryLastTradeResponseMsg
from abides_core.orders import Side
from .trading_agent import TradingAgent
from ..generators import OrderSizeGenerator

logger = logging.getLogger(__name__)

class MovingAverageAgent(TradingAgent):
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
        short_window: int = 20,
        long_window: int = 50,
        strategy: str = "momentum",
    ) -> None:
        super().__init__(id, name, type, random_state, symbol, log_orders, collateral)
        self.wakeup_time: NanosecondTime = wakeup_time
        self.symbol: str = symbol
        self.state: str = "AWAITING_WAKEUP"
        self.size: Optional[int] = (
            self.random_state.randint(20, 50) if order_size_model is None else None
        )
        self.order_size_model = order_size_model
        self.short_window = short_window
        self.long_window = long_window
        self.strategy = strategy.lower()
        assert self.strategy in ["momentum", "mean_reversion"]
        self.trade_history: List[float] = []

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        logger.debug(f"{self.name} initialized with strategy: {self.strategy}")

    def wakeup(self, current_time: NanosecondTime) -> None:
        super().wakeup(current_time)
        self.state = "AWAITING_WAKEUP"
        if not self.mkt_open or not self.mkt_close:
            logger.debug(f"{self.name} detected market is {'open' if self.mkt_open else 'closed'}.")
            return
        if not hasattr(self, 'trading') or not self.trading:
            self.trading = True
            logger.debug(f"{self.name} is ready to start trading.")
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            logger.debug(f"{self.name} detected market closed, stopping trading.")
            return
        if self.wakeup_time and self.wakeup_time > current_time:
            self.set_wakeup(self.wakeup_time)
            logger.debug(f"{self.name} set next wakeup at {self.wakeup_time}.")
            return
        self.get_last_trade(self.symbol)
        self.state = "AWAITING_LAST_TRADE"
        logger.debug(f"{self.name} requested last trade for {self.symbol}.")

    def place_order(self, action: str, price: float) -> None:
        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)
        if self.size and self.size > 0:
            if action == "buy":
                self.place_limit_order(self.symbol, self.size, Side.BID, price)
                logger.info(f"{self.name} placed BUY order at {price} for size {self.size}")
            elif action == "sell":
                self.place_limit_order(self.symbol, self.size, Side.ASK, price)
                logger.info(f"{self.name} placed SELL order at {price} for size {self.size}")

    def receive_message(self, current_time: NanosecondTime, sender_id: int, message: Message) -> None:
        super().receive_message(current_time, sender_id, message)
        if self.state == "AWAITING_LAST_TRADE":
            if isinstance(message, QueryLastTradeResponseMsg):
                last_trade_price = self.last_trade[self.symbol]
                logger.debug(f"{self.name} received last trade price: {last_trade_price}")
                self.trade_history.append(last_trade_price)
                if len(self.trade_history) > self.long_window:
                    removed_price = self.trade_history.pop(0)
                    logger.debug(f"{self.name} removed oldest trade price: {removed_price}")
                if len(self.trade_history) >= self.long_window:
                    self.evaluate_strategy(current_time)
                else:
                    logger.debug(f"{self.name} waiting for more trade data.")
                self.state = "AWAITING_WAKEUP"

    def evaluate_strategy(self, current_time: NanosecondTime) -> None:
        if len(self.trade_history) < self.long_window:
            logger.debug(f"{self.name} waiting for more trade data.")
            return
        prices = self.trade_history[-self.long_window:]
        ma_short = np.mean(prices[-self.short_window:])
        ma_long = np.mean(prices)
        logger.debug(f"{self.name} - MA{self.short_window}: {ma_short}, MA{self.long_window}: {ma_long}")
        last_price = prices[-1]
        if self.strategy == "momentum":
            if ma_short > ma_long:
                logger.debug(f"{self.name} detected uptrend, placing BUY order.")
                self.place_order("buy", last_price)
            elif ma_short < ma_long:
                logger.debug(f"{self.name} detected downtrend, placing SELL order.")
                self.place_order("sell", last_price)
        elif self.strategy == "mean_reversion":
            if ma_short > ma_long:
                logger.debug(f"{self.name} detected overbought, placing SELL order.")
                self.place_order("sell", last_price)
            elif ma_short < ma_long:
                logger.debug(f"{self.name} detected oversold, placing BUY order.")
                self.place_order("buy", last_price)
        next_wakeup = current_time + self.get_wake_frequency()
        self.set_wakeup(next_wakeup)
        logger.debug(f"{self.name} set next wakeup at {next_wakeup}.")

    def get_wake_frequency(self) -> NanosecondTime:
        return self.random_state.randint(low=1_000_000, high=10_000_000)