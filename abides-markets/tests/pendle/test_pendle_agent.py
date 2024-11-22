import logging
import numpy as np
from typing import Optional, Dict, Any, List
import queue
import sys
from abides_core import Message, NanosecondTime
from abides_core.agent import Agent
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from abides_core.utils import str_to_ns, merge_swap, fmt_ts

from datetime import datetime

from abides_core import Message, NanosecondTime
from typing import Any, Dict, List, Optional, Tuple, Type

from abides_core.message import Message, MessageBatch, WakeupMsg, SwapMsg, UpdateRateMsg



# Import the PendleSeedingAgent class from its module
# Adjust the import path based on your project structure
from abides_markets.agents.pendle_agent import PendleSeedingAgent

logger = logging.getLogger(__name__)


class FakeOrderBook:
    def __init__(self):
        self.last_twap = -3.100998
    
    def get_twap(self):
        return self.last_twap
    
    def set_wakeup(self, agent_id: int, requested_time: NanosecondTime) -> None:
        pass
class FakeOracle:
    def __init__(self):
        pass


class FakeKernel:
    def __init__(self,
        agents = {},
        # PENDLE
        swap_interval = str_to_ns("8h"),
        # END PENDLE
        start_time= str_to_ns("00:00:00"),
        ):
        self.agents = {agent.id: agent for agent in agents}
        self.current_time: NanosecondTime = start_time
        self.show_trace_messages: bool = True
        self.messages: queue.PriorityQueue[(int, str, Message)] = queue.PriorityQueue()
        self.book = FakeOrderBook()
        self.rate_normalizer = 1
        self.swap_interval = str_to_ns("8h")
        self.exchange_id = 0
        logger.debug(f"Kernel initialized")
    
    def run(self):
        self.initialize()

        self.runner()

        return self.terminate()
    
    def initialize(self):
        logger.info(f"Simulation started at {fmt_ts(self.current_time)}!")
        logger.debug("--- Agent.kernel_initializing() ---")
        for agent in self.agents:
            agent.kernel_initializing(self)
        logger.debug("--- Agent.kernel_starting() ---")
        for agent in self.agents: 
            agent.kernel_starting(self.start_time)

        # Set the kernel to its start_time.
        self.current_time = self.start_time

        logger.debug("--- Kernel Clock started ---")
        logger.debug("Kernel.current_time is now {}".format(fmt_ts(self.current_time)))

        # Start processing the Event Queue.
        logger.debug("--- Kernel Event Queue begins ---")
        logger.debug(
            "Kernel will start processing messages. Queue length: {}".format(
                len(self.messages.queue)
            )
        )

        # Track starting wall clock time and total message count for stats at the end.
        self.event_queue_wall_clock_start = datetime.now()
        self.ttl_messages = 0

        # PENDLE: 
        # Push message of swaps into message list
        mkt_open, mkt_close = self.agents[0].mkt_open, self.agents[0].mkt_close
        swap_time = mkt_open + self.swap_interval

        while swap_time <= mkt_close:
            for agent in self.agents[1:]:  # Only swap with trading agents
                self.messages.put((swap_time, (-1, agent.id, SwapMsg())))
            swap_time += self.swap_interval

        # END PENDLE
    def send_message(self, sender_id, recipient_id, message, delay=0):
        pass
    
    def runner(
        self, agent_actions: Optional[Tuple[Agent, List[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Start the simulation and processing of the message queue.
        Possibility to add the optional argument agent_actions. It is a list of dictionaries corresponding
        to actions to be performed by the experimental agent (Gym Agent).

        Arguments:
            agent_actions: A list of the different actions to be performed represented in a dictionary per action.

        Returns:
          - it is a dictionnary composed of two elements:
            - "done": boolean True if the simulation is done, else False. It is true when simulation reaches end_time or when the message queue is empty.
            - "results": it is the raw_state returned by the gym experimental agent, contains data that will be formated in the gym environement to formulate state, reward, info etc.. If
               there is no gym experimental agent, then it is None.
        """
        # run an action on a given agent before resuming queue: to be used to take exp agent action before resuming run
        if agent_actions is not None:
            exp_agent, action_list = agent_actions
            exp_agent.apply_actions(action_list)

        # Process messages until there aren't any (at which point there never can
        # be again, because agents only "wake" in response to messages), or until
        # the kernel stop time is reached.
        while (
            not self.messages.empty()
            and self.current_time
            and (self.current_time <= self.stop_time)
        ):
            # Get the next message in timestamp order (delivery time) and extract it.
            self.current_time, event = self.messages.get()
            assert self.current_time is not None

            sender_id, recipient_id, message = event

            # Periodically print the simulation time and total messages, even if muted.
            if self.ttl_messages % 1_000_000 == 0:
                logger.info(
                    "--- Simulation time: {}, messages processed: {:,}, wallclock elapsed: {:.2f}s ---".format(
                        fmt_ts(self.current_time),
                        self.ttl_messages,
                        (
                            datetime.now() - self.event_queue_wall_clock_start
                        ).total_seconds(),
                    )
                )

            if self.show_trace_messages:
                logger.debug("--- Kernel Event Queue pop ---")
                logger.debug(
                    "Kernel handling {} message for agent {} at time {}".format(
                        message.type(), recipient_id, self.current_time
                    )
                )

            self.ttl_messages += 1

            # In between messages, always reset the current_agent_additional_delay.
            self.current_agent_additional_delay = 0

            # Dispatch message to agent.
            if isinstance(message, WakeupMsg):
                # Test to see if the agent is already in the future.  If so,
                # delay the wakeup until the agent can act again.
                if self.agent_current_times[recipient_id] > self.current_time:
                    # Push the wakeup call back into the PQ with a new time.
                    self.messages.put(
                        (
                            self.agent_current_times[recipient_id],
                            (sender_id, recipient_id, message),
                        )
                    )
                    if self.show_trace_messages:
                        logger.debug(
                            "After wakeup return, agent {} delayed from {} to {}".format(
                                recipient_id,
                                fmt_ts(self.current_time),
                                fmt_ts(self.agent_current_times[recipient_id]),
                            )
                        )
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                self.agent_current_times[recipient_id] = self.current_time

                # Wake the agent and get value passed to kernel to listen for kernel interruption signal
                wakeup_result = self.agents[recipient_id].wakeup(self.current_time)

                # Delay the agent by its computation delay plus any transient additional delay requested.
                self.agent_current_times[recipient_id] += (
                    self.agent_computation_delays[recipient_id]
                    + self.current_agent_additional_delay
                )

                if self.show_trace_messages:
                    logger.debug(
                        "After wakeup return, agent {} delayed from {} to {}".format(
                            recipient_id,
                            fmt_ts(self.current_time),
                            fmt_ts(self.agent_current_times[recipient_id]),
                        )
                    )
                # catch kernel interruption signal and return wakeup_result which is the raw state from gym agent
                if wakeup_result != None:
                    return {"done": False, "result": wakeup_result}

            # PENDLE: Detect Swap Msg
            elif isinstance(message, SwapMsg):
                # Test to see if the agent is already in the future.  If so,
                # delay the swap until the agent can act again.
                if self.agent_current_times[recipient_id] > self.current_time:
                    # Push the wakeup call back into the PQ with a new time.
                    self.messages.put(
                        (
                            self.agent_current_times[recipient_id],
                            (sender_id, recipient_id, message),
                        )
                    )
                    if self.show_trace_messages:
                        logger.debug(
                            "After swap return, agent {} delayed from {} to {}".format(
                                recipient_id,
                                fmt_ts(self.current_time),
                                fmt_ts(self.agent_current_times[recipient_id]),
                            )
                        )
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                self.agent_current_times[recipient_id] = self.current_time

                # Swap the agent and get value passed to kernel to listen for kernel interruption signal
                swap_result = self.agents[recipient_id].swap(self.current_time, self.rate_oracle.get_floating_rate(self.current_time))

                # Delay the agent by its computation delay plus any transient additional delay requested.
                self.agent_current_times[recipient_id] += (
                    self.agent_computation_delays[recipient_id]
                    + self.current_agent_additional_delay
                )

                if self.show_trace_messages:
                    logger.debug(
                        "After swap return, agent {} delayed from {} to {}".format(
                            recipient_id,
                            fmt_ts(self.current_time),
                            fmt_ts(self.agent_current_times[recipient_id]),
                        )
                    )
                # catch kernel interruption signal and return swap_result
                if swap_result != None:
                    return {"done": False, "result": swap_result}
                
            # END PENDLE
            else:
                # Test to see if the agent is already in the future.  If so,
                # delay the message until the agent can act again.
                if self.agent_current_times[recipient_id] > self.current_time:
                    # Push the message back into the PQ with a new time.
                    self.messages.put(
                        (
                            self.agent_current_times[recipient_id],
                            (sender_id, recipient_id, message),
                        )
                    )
                    if self.show_trace_messages:
                        logger.debug(
                            "Agent in future: message requeued for {}".format(
                                fmt_ts(self.agent_current_times[recipient_id])
                            )
                        )
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                self.agent_current_times[recipient_id] = self.current_time

                # Deliver the message.
                if isinstance(message, MessageBatch):
                    messages = message.messages
                else:
                    messages = [message]

                for message in messages:
                    # Delay the agent by its computation delay plus any transient additional delay requested.
                    self.agent_current_times[recipient_id] += (
                        self.agent_computation_delays[recipient_id]
                        + self.current_agent_additional_delay
                    )

                    if self.show_trace_messages:
                        logger.debug(
                            "After receive_message return, agent {} delayed from {} to {}".format(
                                recipient_id,
                                fmt_ts(self.current_time),
                                fmt_ts(self.agent_current_times[recipient_id]),
                            )
                        )

                    self.agents[recipient_id].receive_message(
                        self.current_time, sender_id, message
                    )

        if self.messages.empty():
            logger.info("--- Kernel Event Queue empty ---")

        if self.current_time and (self.current_time > self.stop_time):
            logger.info(f"--- Kernel Stop Time {self.stop_time} surpassed ---")

        return {"done": True, "result": None}
def test_pendle_seeding_agent():
    """
    Test the PendleSeedingAgent to ensure it correctly seeds the market when the order book is empty.
    """
    module_name = PendleSeedingAgent.__module__
    logger.debug(f"PendleSeedingAgent imported from module: {module_name}")

    # Get the module object
    module = sys.modules[module_name]

    # Get the file location
    file_location = module.__file__
    logger.debug(f"PendleSeedingAgent file location: {file_location}")

    logger.debug("Starting test_pendle_seeding_agent")

    # Initialize the agent with specific parameters
    agent_id = 1
    symbol = "PEN"
    size = 100
    min_bid = 1
    max_bid = 10
    min_ask = 11
    max_ask = 20

    logger.debug(f"Initializing PendleSeedingAgent with ID {agent_id} and symbol '{symbol}'")
    logger.debug(f"Order size: {size}, Bid price range: {min_bid}-{max_bid}, Ask price range: {min_ask}-{max_ask}")

    # Create a random state for reproducibility
    random_state = np.random.RandomState(seed=42)
    logger.debug("Created random state for reproducibility")

    # Instantiate the PendleSeedingAgent
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
    logger.debug("PendleSeedingAgent instantiated")

    # Assign the kernel to the agent
    kernel = FakeKernel()
    seeding_agent.kernel = kernel
    kernel.agents[seeding_agent.id] = seeding_agent
    logger.debug("Assigned FakeKernel to the agent")

    # Set market open and close times
    seeding_agent.mkt_open = 1
    seeding_agent.mkt_close = 1_000_000_000  # Arbitrary close time
    seeding_agent.current_time = 0
    logger.debug(f"Set market open time to {seeding_agent.mkt_open} and close time to {seeding_agent.mkt_close}")

    # Set the exchange ID (assuming it's 0)
    seeding_agent.exchange_id = 0
    logger.debug(f"Set exchange ID to {seeding_agent.exchange_id}")

    # Mock known bids and asks to simulate an empty order book
    seeding_agent.known_bids = {symbol: []}
    seeding_agent.known_asks = {symbol: []}
    logger.debug("Simulated empty order book by setting known bids and asks to empty lists")

    # Mock methods to capture placed orders
    placed_orders = []
    logger.debug("Initialized list to capture placed orders")

    def mock_place_limit_order(symbol, quantity, side, price):
        logger.debug(f"Placed limit order - Symbol: {symbol}, Quantity: {quantity}, Side: {side}, Price: {price}")
        placed_orders.append({'symbol': symbol, 'quantity': quantity, 'side': side, 'price': price})

    seeding_agent.place_limit_order = mock_place_limit_order
    logger.debug("Replaced agent's place_limit_order method with mock method")

    # Simulate the agent's wakeup call
    logger.debug("Simulating agent's wakeup call")
    seeding_agent.wakeup(seeding_agent.current_time)
    logger.debug(f"state: {seeding_agent.state},type: {seeding_agent.type}")

    # Simulate receiving a QuerySpreadResponseMsg indicating an empty order book
    logger.debug("Simulating receipt of QuerySpreadResponseMsg with empty order book")
    message = QuerySpreadResponseMsg(
        symbol=symbol,
        bids=[],
        asks=[],
        mkt_closed=False,
        depth=0,
        last_trade=None,
    )

    # The agent receives the message and should proceed to seed the market
    logger.debug("Agent receiving the QuerySpreadResponseMsg")
    seeding_agent.receive_message(seeding_agent.current_time, sender_id=0, message=message)

    # Verify that the agent placed the correct orders
    expected_bid_prices = list(range(min_bid, max_bid + 1))
    expected_ask_prices = list(range(min_ask, max_ask + 1))
    logger.debug(f"Expected bid prices: {expected_bid_prices}")
    logger.debug(f"Expected ask prices: {expected_ask_prices}")

    # Separate the placed bids and asks
    placed_bids = [order for order in placed_orders if order['side'] == Side.BID]
    placed_asks = [order for order in placed_orders if order['side'] == Side.ASK]
    logger.debug(f"Number of bids placed: {len(placed_bids)}")
    logger.debug(f"Number of asks placed: {len(placed_asks)}")

    # Check that the correct number of bids and asks were placed
    assert len(placed_bids) == len(expected_bid_prices), \
        f"Expected {len(expected_bid_prices)} bids, got {len(placed_bids)}"
    logger.debug("Correct number of bids placed")

    assert len(placed_asks) == len(expected_ask_prices), \
        f"Expected {len(expected_ask_prices)} asks, got {len(placed_asks)}"
    logger.debug("Correct number of asks placed")

    # Verify the bid orders
    for price in expected_bid_prices:
        if any(order['price'] == price and order['quantity'] == size for order in placed_bids):
            logger.debug(f"Verified bid order at price {price}")
        else:
            logger.error(f"Bid at price {price} not found")
            assert False, f"Bid at price {price} not found"

    # Verify the ask orders
    for price in expected_ask_prices:
        if any(order['price'] == price and order['quantity'] == size for order in placed_asks):
            logger.debug(f"Verified ask order at price {price}")
        else:
            logger.error(f"Ask at price {price} not found")
            assert False, f"Ask at price {price} not found"

    logger.info("PendleSeedingAgent test passed successfully.")