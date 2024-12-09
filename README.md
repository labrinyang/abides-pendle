<div id="top"></div>

# ABIDES: Agent-Based Interactive Discrete Event Simulation environment

<!-- TABLE OF CONTENTS -->
<ol>
  <li>
    <a href="#about-the-project">About The Project</a>
  </li>
  <li>
    <a href="#getting-started">Getting Started</a>
  </li>
  <li>
    <a href="#usage">Usage</a>
  </li>
  <li><a href="#markets-configurations">Markets Configurations</a></li>
  <li><a href="#agent-configurations">Agents Configurations</a></li>
</ol>

<!-- ABOUT THE PROJECT -->
## About The Project

This event simulator is based on ABIDES (Agent Based Interactive Discrete Event Simulator). This is specifically designed to simulate Pendle markets. 

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
### Installation

1. Download the source code.

2. Run the install script to install the packages and their dependencies:

    ```
    sh install.sh
    ```


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage (regular)
Regular ABIDES simulations can be run directly in python. The sample is in `notebooks/demo_PENDLE.ipynb`.

```python
from abides_markets.configs import prmsc1, prmsc2
from abides_core import abides

config_state = prmsc02.build_config()
end_state = abides.run(config_state)
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Markets Configurations

The repo currently has the following available background Market Simulation Configuration:

* PRMSC01: 1 Exchange Agent, 2 Market Maker Agent, 5 Value Agents, 10 Noise Agents, 1 Liquidator Agents (primarily for testing)
 
* PRMSC02: 1 Exchange Agent, 2 Market Maker Agents, 100 Value Agents, 1000 Noise Agents, 1 Liquidator Agents. For details and modifications, go to `abides-markets/abides_markets/configs/prmsc2.py`.

The order size model: each order have notional size of $1M. 10% of the time, the order size gets higher, up to $10M. For details and modifications, go to `abides-markets/abides_markets/models/order_size_model.py`. 

There are 2 settings of driving oracle ready to be used. Details in `abides-markets/abides_markets/driving_oracle`.

* Manual Oracle: Mean-reversion around a constant mean with some predefined megashock applied through out the simulation duration.

* Linear Oracle: Mean-reversion around a mean which is a linear function of time, predefined in construction. 

There are 2 settings of rate oracle ready to be used. Details in `abides-markets/abides_markets/rate_oracle`.

* Constant Oracle: An oracle gives a constant funding rate.

* BTC Oracle: An oracle gives the true funding rate of BTCUSDT in Binance in 2023.
<p align="right">(<a href="#top">back to top</a>)</p>

## Agents Configurations

* Market Maker: Record transaction volume within its average sleep time (e.g. it wakes up with rate 10 minutes, then everytime it wakes, it record the transaction volume in the most recent 10 minutes). Place 10 bid orders and 10 ask orders, distance 0.25%, total size = recorded transaction volume. Order size is linearly proportional to the distance to `mid_price` (the closer tick, the bigger size).  


<p align="right">(<a href="#top">back to top</a>)</p>
