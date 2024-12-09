import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from abides_markets.rate_oracle import ConstantOracle
from abides_markets.rate_oracle import BTCOracle
from abides_markets.driving_oracle import ManualOracle, LinearOracle
from abides_markets.agents import ValueAgent

from abides_core import Kernel
from abides_core.utils import datetime_str_to_ns, str_to_ns
from abides_markets.driving_oracle import ManualOracle, LinearOracle
from abides_markets.agents import ValueAgent

from abides_core import Kernel
from abides_core.utils import datetime_str_to_ns, str_to_ns


def test_const_oracle():
    const_oracle = ConstantOracle(0, 0, 10)

    a = const_oracle.get_floating_rate(12345)
    b = const_oracle.get_floating_rate(12345)

    assert a == b

def test_BTC_oracle():
    oracle = BTCOracle()
    assert round(oracle.get_floating_rate(1_673_222_400_003_000_001) - 5.796000e-05, 10) == 0
    assert oracle.get_floating_rate(1_675_238_400_000_000_000) == 0.0001
    
    assert round(oracle.get_floating_rate(1_673_222_400_003_000_001) - 5.796000e-05, 10) == 0
    assert oracle.get_floating_rate(1_675_238_400_000_000_000) == 0.0001
    
    assert True

def test_manual_oracle():
    MKT_OPEN = int(pd.to_datetime("20230101").to_datetime64())
    MKT_CLOSE = int(pd.to_datetime("20230201").to_datetime64())
    swap_interval = str_to_ns("8h")
    ticker = "PEN"

    # Manual Oracle
    symbols = {
        ticker: {
            "r_bar": 0.10,
            "kappa": 0.001,
            "sigma_s": 100
        }
    }

    manual_oracle = ManualOracle(MKT_OPEN, MKT_CLOSE, symbols, [{"time": 1/2, "mag": 1000}])
    oracle_value = manual_oracle.r[ticker]
    X = np.linspace(0, len(oracle_value)-1, 31*12+1).astype(int)

    assert len(oracle_value) == 31*24*60+1
    observed_price = np.zeros(len(X))

    for i, j in enumerate(X):
        observed_price[i] = manual_oracle.observe_price(ticker, MKT_OPEN + str_to_ns("1min")*j, np.random.RandomState(i), sigma_n=100**2)

    plt.figure()
    plt.plot(X, [oracle_value[x] for x in X], label='True value from Oracle', linewidth=2)
    plt.plot(X, observed_price, label='Noisy observation from Oracle', linewidth=1)
    plt.legend()

    plt.savefig('manual_oracle_plot.png')

def test_linear_oracle():
    MKT_OPEN = int(pd.to_datetime("20230101").to_datetime64())
    MKT_CLOSE = int(pd.to_datetime("20230201").to_datetime64())
    swap_interval = str_to_ns("8h")
    ticker = "PEN"

    # Linear Oracle
    symbols = {
        ticker: {
            "kappa": 0.01,
            "sigma_s": 10**2
        }
    }

    linear_oracle = LinearOracle(MKT_OPEN, MKT_CLOSE, symbols, [{"time": 0, "mag": 1000},
                                                                {"time": 1/3, "mag": 1500},
                                                                {"time": 2/3, "mag": 500},
                                                                {"time": 1, "mag": 1000}
                                                                ])
    oracle_value = linear_oracle.r[ticker]
    X = np.linspace(0, len(oracle_value)-1, 31*12+1).astype(int)

    assert len(oracle_value) == 31*24*60+1
    observed_price = np.zeros(len(X))

    for i, j in enumerate(X):
        observed_price[i] = linear_oracle.observe_price(ticker, MKT_OPEN + str_to_ns("1min")*j, np.random.RandomState(i), sigma_n=100**2)

    plt.figure()
    plt.plot(X, [oracle_value[x] for x in X], label='True value from Oracle', linewidth=2)
    plt.plot(X, observed_price, label='Noisy observation from Oracle', linewidth=1)
    plt.legend()

    plt.savefig('linear_oracle_plot.png')

    ##

    value_agent = ValueAgent(id=0)
    value_agent.linear_oracle = linear_oracle
    value_agent.rate_oracle = ConstantOracle()
    kernel = Kernel([value_agent], 
                    swap_interval = swap_interval,
                    )
    value_agent.kernel = kernel

    # value_agent.current_time = current_time

    # value_agent.updateEstimates()

    assert value_agent.r_t

    