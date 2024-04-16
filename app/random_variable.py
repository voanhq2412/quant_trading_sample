from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.max_rows", None)


class RandomVariable:
    def __init__(self, data: np.array, dist: stats._continuous_distns):
        self.dist = dist
        self.data = data
        self.dist_params = self.dist.fit(self.data)
        self.count_fit = 0
        self.ALPHA = [0, 0.05, 0.1, 0.4, 0.6, 0.9, 0.95, 1]

    def mle(self) -> float:
        return np.sum(np.log(self.dist.pdf(self.data, *self.dist_params)))

    def transform(self, uniform_data: np.array) -> np.array:
        return self.dist.ppf(uniform_data, *self.dist_params)

    def simulate(self, sample_size: int) -> np.array:
        return self.dist.rvs(*self.dist_params, size=sample_size)

    def get_quantiles(self, data: np.array) -> Dict[float, float]:
        quantiles = {a: np.quantile(data, a) for a in self.ALPHA}
        return quantiles

    # gen_hyperbolic can overfit,
    # transforming could lead to values lying outside [0,1] range
    def unif(self) -> np.array:
        self.count_fit += 1
        u = self.dist.cdf(self.data, *self.dist_params)
        return u
