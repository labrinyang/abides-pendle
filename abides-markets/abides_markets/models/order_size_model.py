import json

import numpy as np
from pomegranate import GeneralMixtureModel


order_size = {
    "class": "GeneralMixtureModel",
    "distributions": [
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [0.15, 0.02],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [0.60, 0.06],
            "frozen": True,
        },
    ],
    "weights": [
        0.8,
        0.2
    ],
}


class OrderSizeModel:
    def __init__(self) -> None:
        self.model = GeneralMixtureModel.from_json(json.dumps(order_size))

    def sample(self, random_state: np.random.RandomState) -> float:
        return self.model.sample(random_state=random_state)
    
class NormalOrderSizeModel:
    def __init__(self, mean, sd) -> None:
        self.mean = mean
        self.sd = sd

    def sample(self, random_state: np.random.RandomState) -> float:
        return np.random.normal(self.mean, self.sd)
