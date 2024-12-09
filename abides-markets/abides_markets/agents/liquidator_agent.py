from typing import List, Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns, merge_swap

from ..messages.marketdata import MarketDataMsg, L2SubReqMsg
from ..messages.query import QuerySpreadResponseMsg
from ..orders import Side
from .trading_agent import TradingAgent, ExchangeAgent


class LiquidatorAgent(TradingAgent):
    """
    Trading Agent that wakes up occasionally, check every other trading agents 
    for their maintainance margin and liquidate them if possible.
    """

    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        symbol: str = "PEN",
        wake_up_freq: NanosecondTime = str_to_ns("1h"),
        subscribe=False,
        log_orders=False,
        collateral: float = 100_000,
        watch_list: List[int] = []
    ) -> None:

        super().__init__(id, name, type, random_state, symbol, log_orders, collateral)
        self.wake_up_freq = wake_up_freq

        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscription_requested = False
        self.log_orders = log_orders
        self.watch_list = watch_list  # List of agents to watch for liquidation
        self.state = "AWAITING_WAKEUP"
        self.failed_liquidation = 0

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

        self.watch_list: List[int] = self.kernel.find_agents_by_type(TradingAgent)

    def wakeup(self, current_time: NanosecondTime) -> None:
        """Agent wakeup is determined by self.wake_up_freq"""
        can_trade = super().wakeup(current_time)
        if self.subscribe and not self.subscription_requested:
            super().request_data_subscription(
                L2SubReqMsg(
                    symbol=self.symbol,
                    freq=int(10e9),
                    depth=10,
                )
            )
            self.subscription_requested = True
            self.state = "AWAITING_MARKET_DATA"
        elif can_trade and not self.subscribe:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)
        if (
            not self.subscribe
            and self.state == "AWAITING_SPREAD"
            and isinstance(message, QuerySpreadResponseMsg)
        ):
            if self.position["SIZE"] != 0:  # Only continue liquidate after sell position
                self.set_wakeup(current_time + self.get_quick_wake_frequency())
                self.logEvent("WAIT_FOR_SELL", self.position["SIZE"])
            else: 
                liquidated = False

                for agent_id in self.watch_list:
                    if self.check_liquidate(self.kernel.agents[agent_id]):  # It should be in term of msg rather than directly like this
                        liquidated = True
                        break  # Only liquidate one agent each time
                
                if liquidated: 
                    self.set_wakeup(current_time + self.get_quick_wake_frequency())
                else:
                    self.logEvent("NO_LIQUIDATION")
                    self.set_wakeup(current_time + self.get_wake_frequency())

                self.state = "AWAITING_WAKEUP"

        elif (  # TODO: Listen to every order
            self.subscribe
            and self.state == "AWAITING_MARKET_DATA"
            and isinstance(message, MarketDataMsg)
        ):
            for agent_id in self.watch_list:
                self.check_liquidate(self.kernel.agents[agent_id])

            self.state = "AWAITING_MARKET_DATA"

    def check_liquidate(self, agent: TradingAgent, sell: bool = True) -> bool:
        """
        Check if an agent is liquidatable. Liquidate him if profitable. Return True if liquidate an amount successfully. 
        
        Arguments:
            agent: the agent to be liquidated
            sell: whether or not sell the position immediately after liquidation 
        """
        agent.current_time = self.current_time  # Because this function is not in term of msg

        if agent.is_healthy() or agent.position["SIZE"] == 0:
            return False

        self.logEvent("LIQUIDATABLE", f"AGENT ID: {agent.id} R1 = {agent.last_R1}")

        mRatio = agent.mRatio()

        if mRatio > 1:
            return False
        assert mRatio >=0 and mRatio <= 1, mRatio

        liq_fac_base = 0.1
        liq_fac_slope = 1
        liq_ict_fact = liq_fac_base + liq_fac_slope * (1 - mRatio)

        agent.cancel_all_orders()

        market_tick = self.kernel.book.last_twap
        longing = True if agent.position["SIZE"] > 0 else False  # indicate agent is longing yield
        d_size = 0

        if longing:  # We need to liquidate by market ask order (selling) (i.e. look at BID wall)
            for bid in self.known_bids[self.symbol]:
                if bid[0] < market_tick:
                    break

                d_size += bid[1]

                if d_size >= agent.position["SIZE"]:
                    d_size = agent.position["SIZE"]
                    break
            
        else:  # We need to liquidate by market bid order (buying) (i.e. look at ASK wall)
            for ask in self.known_asks[self.symbol]:
                if ask[0] > market_tick:
                    break
                d_size -= ask[1]

                if d_size <= agent.position["SIZE"]:
                    d_size = agent.position["SIZE"]
                    break
            
        l = d_size/agent.position["SIZE"]
        new_position_size = (1-l) * agent.position["SIZE"]

        assert l >= 0 and l <= 1
        agent.logN1(abs(new_position_size))

        if l == 0:
            self.logEvent("FAILED_LIQUIDATION", f"AGENT ID: {agent.id}")
            return False

        # Transfer the collateral
        p_unrealized = agent.mark_to_market() - agent.position["COLLATERAL"]
        liq_val = l*p_unrealized
        
        marginDelta = agent.maintainance_margin() - agent.maintainance_margin(new_position_size)
        liq_incentive = min(liq_ict_fact, mRatio) * marginDelta
        d_col = -liq_val + liq_incentive
        
        agent.position["COLLATERAL"] -= d_col
        self.position["COLLATERAL"] += d_col

        # Transfer the position
        assert self.position["SIZE"] == 0, self.position["SIZE"]
        
        self.position["SIZE"], self.position["FIXRATE"], p_merge_pa = merge_swap(self.position["SIZE"], self.position["FIXRATE"], 
                                                                        d_size, agent.position["FIXRATE"])
        
        assert p_merge_pa == 0

        agent.position["SIZE"] -= d_size
        if agent.position["SIZE"] == 0:
            agent.position["FIXRATE"] == 0

        self.logEvent("SUCCESSFUL_LIQUIDATION", 
                      f"AGENT ID: {agent.id}, SIZE {d_size}")
        
        self.position_updated()
        agent.liquidated(d_col, d_size)  # Also called position_updated here

        # Sell the position immediately
        if sell:
            if d_size > 0:
                self.place_market_order(self.symbol, d_size, Side.ASK)
            else:
                self.place_market_order(self.symbol, -d_size, Side.BID)

        # TODO: Update internal known_bids/asks for the next loop

        return True

    def get_wake_frequency(self) -> NanosecondTime:
        return self.wake_up_freq
