import numpy as np
from xlrd.sheet import Sheet
from sabr_calibrator import lazy_property


class SABRDataLoader:

    def __init__(self, market_data):
        assert isinstance(market_data, Sheet)
        self.market_data = market_data

    @lazy_property
    def strike_spreads(self):
        # ----- read strike spreads
        strike_spreads = []
        j = 0
        while True:
            try:
                strike_spreads.append(int(self.market_data.cell(1, 3 + j).value))
                j = j + 1
            except IndexError:
                break
        return strike_spreads

    @lazy_property
    def num_strikes(self):
        return len(self.strike_spreads)

    @lazy_property
    def expiries(self):
        # ----- read times to expiry
        expiries = []
        i = 0
        while True:
            try:
                expiries.append(self.market_data.cell(2 + i, 1).value)
                i = i + 1
            except IndexError:
                break
        return expiries

    @lazy_property
    def tenors(self):
        # ----- read tenors
        tenors = []
        i = 0
        while True:
            try:
                tenors.append(self.market_data.cell(2 + i, 0).value)
                i = i + 1
            except IndexError:
                break
        return tenors

    @lazy_property
    def frates(self):
        # ----- read forward rates
        frates = []
        i = 0
        while True:
            try:
                frates.append(self.market_data.cell(2 + i, 2).value)
                i = i + 1
            except IndexError:
                break
        return frates

    @lazy_property
    def mkt_vols(self):
        # ----- read market volatilities
        mkt_vols = np.zeros((len(self.frates), self.num_strikes))
        for i in range(len(self.frates)):
            for j in range(self.num_strikes):
                mkt_vols[i][j] = self.market_data.cell(2 + i, 3 + j).value
        return mkt_vols

    @lazy_property
    def strike_grid(self):
        # ----- create the strikes' grid
        strike_grid = np.zeros((len(self.frates), self.num_strikes))
        for i in range(len(self.frates)):
            for j in range(self.num_strikes):
                strike_grid[i][j] = self.frates[i] + 0.0001 * (self.strike_spreads[j])
        return strike_grid
