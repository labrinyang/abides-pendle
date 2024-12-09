import numpy as np

from abides_core.kernel import Kernel
from abides_markets.agents import TradingAgent, LiquidatorAgent
from abides_core.utils import str_to_ns
from abides_markets.agents.utils import tick_to_rate

from abides_markets.rate_oracle import ConstantOracle


def test_maintainance_margin():
    agent = TradingAgent(id=0)
    agent.rate_oracle = ConstantOracle()
    kernel = Kernel([agent], 
                    swap_interval = str_to_ns("8h"),
                    )

    agent.kernel = kernel

    agent.position = {"COLLATERAL": 100,
                      "SIZE": 0,
                      "FIXRATE": 0}

    agent.n_payment = 365*3
    agent.rate_normalizer = 1/(365*3)

    test_size_thresh = [20, 100]
    test_mm_fac = [0.03, 0.06, 0.1]

    assert agent.maintainance_margin(None, test_size_thresh, test_mm_fac) == 0
    assert agent.maintainance_margin(10, test_size_thresh, test_mm_fac) == 0.3
    assert agent.maintainance_margin(20, test_size_thresh, test_mm_fac) == 0.6
    assert agent.maintainance_margin(60, test_size_thresh, test_mm_fac) == 3
    assert round(agent.maintainance_margin(110, test_size_thresh, test_mm_fac), 1) == 6.4

class FakeOrderBook:
    def __init__(self):
        self.last_twap = 1000
    
class FakeOracle:
    def __init__(self):
        pass

def test_mark_to_market():
    agent = TradingAgent(id=0)

    kernel = Kernel([agent], 
                    swap_interval = str_to_ns("8h"),
                    )
    kernel.book = FakeOrderBook()

    agent.kernel = kernel

    agent.position = {"COLLATERAL": 100,
                      "SIZE": 100,
                      "FIXRATE": 0.20}
    
    agent.n_payment = 365*3
    agent.rate_normalizer = 1/(365*3)

    assert agent.mark_to_market(log=False) == 100 + 100 * (tick_to_rate(1000) - 0.20)       
    

def test_liquidation_status():
    agent = TradingAgent(id=0)

    kernel = Kernel([agent], 
                    swap_interval = str_to_ns("8h"),
                    )
    kernel.book = FakeOrderBook()

    agent.kernel = kernel
    agent.n_payment = 365*3
    agent.rate_normalizer = 1/(365*3)

    agent.position = {"COLLATERAL": 20,
                      "SIZE": 100,
                      "FIXRATE": 0.20}
    
    agent.position_updated()

    assert round(agent.mark_to_market(log=False) - (20 + 100 * (tick_to_rate(1000) - 0.20)), 4) == 0
    assert round(agent.maintainance_margin(100) - 3.8, 4) == 0

    assert round(agent.mRatio() - (20 + 100*(tick_to_rate(1000) - 0.20))/3.8, 4) == 0
    assert agent.is_healthy()

    agent.position = {"COLLATERAL": 10,
                      "SIZE": 100,
                      "FIXRATE": 0.20}
    agent.position_updated()
    
    assert not agent.is_healthy()

def test_liquidator():
    liquidator = LiquidatorAgent(id=0)
    dummy_agent = TradingAgent(id=1)

    kernel = Kernel([liquidator, dummy_agent], 
                    swap_interval = str_to_ns("8h"),
                    )
    kernel.book = FakeOrderBook()  # Book return twap = 10%

    dummy_agent.kernel = kernel
    liquidator.kernel = kernel

    dummy_agent.position = {"COLLATERAL": 10,
                            "SIZE": 100,
                            "FIXRATE": 0.20}
    
    dummy_agent.n_payment = 365*3
    liquidator.n_payment = 365*3
    dummy_agent.rate_normalizer = 1/(365*3)
    liquidator.rate_normalizer = 1/(365*3)
    
    liquidator.known_bids[liquidator.symbol] = [
        [1100, 10],  
        [1000, 10],
        [950, 10]
    ]

    # Liquidator is supposed to liquidate agent by taking (SIZE:20, FIXRATE:0.20) position

    liquidator.check_liquidate(dummy_agent, sell=False)
    assert liquidator.position["SIZE"] == 20

    assert dummy_agent.position["SIZE"] == 80
    # assert dummy_agent.position["COLLATERAL"] > 0
    # assert dummy_agent.position["COLLATERAL"] < 14



    



