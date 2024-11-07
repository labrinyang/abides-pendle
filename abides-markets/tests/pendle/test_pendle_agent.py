import unittest
from unittest.mock import Mock
from abides_core import NanosecondTime
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from abides_markets.agents import PendleSeedingAgent
import numpy as np

class TestPendleSeedingAgent(unittest.TestCase):

    def setUp(self):
        self.agent = PendleSeedingAgent(
            id=1,
            name="PendleSeedingAgent",
            random_state=np.random.RandomState(42)
        )
        self.agent.kernel = Mock()
        self.agent.kernel.pen_oracle = Mock()
        self.agent.mkt_open = True
        self.agent.mkt_close = False
        self.agent.mkt_closed = False
        self.agent.exchange_id = 2

    def test_initialization(self):
        self.assertEqual(self.agent.state, "AWAITING_WAKEUP")
        self.assertEqual(self.agent.symbol, "PEN")
        self.assertEqual(self.agent.size, 100)
        self.assertEqual(self.agent.min_bid, 1)
        self.assertEqual(self.agent.max_bid, 10)
        self.assertEqual(self.agent.min_ask, 11)
        self.assertEqual(self.agent.max_ask, 20)

    def test_wakeup_to_awaiting_spread(self):
        self.agent.wakeup(current_time=NanosecondTime(0))
        self.assertEqual(self.agent.state, "AWAITING_SPREAD")

    def test_receive_query_spread_response(self):
        self.agent.state = "AWAITING_SPREAD"
        message = QuerySpreadResponseMsg()
        self.agent.get_known_bid_ask = Mock(return_value=(None, None, None, None))
        self.agent.seed = Mock()

        self.agent.receive_message(current_time=NanosecondTime(0), sender_id=2, message=message)

        self.agent.seed.assert_called_once()
        self.assertEqual(self.agent.state, "AWAITING_WAKEUP")

    def test_seed_places_orders_in_price_range(self):
        self.agent.place_limit_order = Mock()
        self.agent.seed()

        for price in range(self.agent.min_bid, self.agent.max_bid + 1):
            self.agent.place_limit_order.assert_any_call(
                self.agent.symbol, self.agent.size, Side.BID, price
            )

        for price in range(self.agent.min_ask, self.agent.max_ask + 1):
            self.agent.place_limit_order.assert_any_call(
                self.agent.symbol, self.agent.size, Side.ASK, price
            )

    def test_no_orders_if_market_closed(self):
        self.agent.mkt_closed = True
        self.agent.place_limit_order = Mock()
        self.agent.seed()

        self.agent.place_limit_order.assert_not_called()

if __name__ == "__main__":
    unittest.main()
