import numpy as np
from scipy.optimize import minimize
from sabr_calibrator import SABRDataLoader, lazy_property
import warnings
warnings.filterwarnings(action='ignore')


class SABRCalibrator:

    def __init__(self, starting_values, market_data):
        self.starting_values = starting_values
        self.data_loader = SABRDataLoader(market_data)

    @lazy_property
    def calibrated_params(self):
        alphas, betas, rhos, nus = list(), list(), list(), list()
        x0 = self.starting_values
        bnds = ((0.001, None), (0, 1), (-0.999, 0.999), (0.001, None))
        for frate, strike, time, mkt_vol in zip(self.data_loader.frates, self.data_loader.strike_grid,
                                                self.data_loader.expiries, self.data_loader.mkt_vols):
            # constrained optimization
            res = minimize(self.objective_function, x0, (frate, strike, time, mkt_vol),
                           bounds=bnds, method='trust-constr')
            alphas.append(res.x[0])
            betas.append(res.x[1])
            rhos.append(res.x[2])
            nus.append(res.x[3])
        return alphas, betas, rhos, nus

    def objective_function(self, params, f, strikes, time_ex, mkt_vols):
        # alpha: params[0]
        # beta: params[1]
        # rho: params[2]
        # nu: params[3]
        sum_sq_diff = 0
        if strikes[0] <= 0:
            f, strikes = self._shift_f_strikes(f, strikes)
        for strike, mkt_vol in zip(strikes, mkt_vols):
            if f == strike:  # ATM formula from Hagan et al.
                factor_1 = params[0] / (f ** (1 - params[1]))
                factor_2 = 1 + ((1 - params[1]) ** 2 / 24 * (factor_1 ** 2) + (
                            params[1] * params[3] * params[2]) / 4 * factor_1 + (
                                        params[3] ** 2 * (2 - 3 * params[2] ** 2) / 24)) * time_ex
                model_vol = factor_1 * factor_2
                diff = model_vol - mkt_vol
            else:  # non-ATM formula from Hagan et al.
                factor_1 = params[0] / ((f * strike) ** ((1 - params[1]) / 2))
                log_fk = np.log(f / strike)
                z = params[3] / factor_1 * log_fk
                x = np.log((np.sqrt(1 - 2 * params[2] * z + z ** 2) + z - params[2]) / (1 - params[2]))
                factor_2 = 1 / (1 + (1 / 24) * (((1 - params[1]) * log_fk) ** 2) + (1 / 1920) * (
                            ((1 - params[1]) * log_fk) ** 4))
                factor_3 = z / x
                factor_4 = 1 + ((1 - params[1]) ** 2 / 24 * (factor_1 ** 2) + (
                            params[1] * params[3] * params[2]) / 4 * factor_1 + (
                                        params[3] ** 2 * (2 - 3 * params[2] ** 2) / 24)) * time_ex
                model_vol = factor_1 * factor_2 * factor_3 * factor_4
                diff = model_vol - mkt_vol
            sum_sq_diff = sum_sq_diff + diff ** 2
        objfunc = np.sqrt(sum_sq_diff)
        return objfunc

    @lazy_property
    def sabr_vol_matrix(self):
        alphas, betas, rhos, nus = self.calibrated_params
        sabr_smiles = list()
        for alpha, beta, rho, nu, frate, strikes, expiry in zip(alphas, betas, rhos, nus,
                                                                self.data_loader.frates,
                                                                self.data_loader.strike_grid,
                                                                self.data_loader.expiries):
            sabr_smile = self._get_smile(alpha, beta, rho, nu, frate, strikes, expiry)
            sabr_smiles.append(sabr_smile)
        return np.array(sabr_smiles)

    @classmethod
    def _get_smile(cls, alpha, beta, rho, nu, f, strikes, time):
        if strikes[0] <= 0:
            f, strikes = cls._shift_f_strikes(f, strikes)
        vget_sabr_vol = np.vectorize(cls._get_sabr_vol)
        sabr_vol_smile = vget_sabr_vol(alpha, beta, rho, nu, f, strikes, time)
        return sabr_vol_smile

    @staticmethod
    def _shift_f_strikes(f, strikes):
        shift = 0.001 - strikes[0]
        strikes_shifted = list()
        for strike in strikes:
            strikes_shifted.append(strike + shift)
        f_new = f + shift
        return f_new, np.array(strikes_shifted)

    @staticmethod
    def _get_sabr_vol(alpha, beta, rho, nu, f, strike, time_ex):  # all variables are scalars
        if f == strike:  # ATM formula from Hagan et al.
            factor_1 = alpha / (f ** (1 - beta))
            factor_2 = 1 + ((1 - beta) ** 2 / 24 * (factor_1 ** 2) + (beta * nu * rho) / 4 * factor_1 + (
                    nu ** 2 * (2 - 3 * rho ** 2) / 24)) * time_ex
            model_vol = factor_1 * factor_2
        else:  # non-ATM formula from Hagan et al.
            factor_1 = alpha / ((f * strike) ** ((1 - beta) / 2))
            log_fk = np.log(f / strike)
            z = nu / factor_1 * log_fk
            x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
            factor_2 = 1 / (1 + (1 / 24) * (((1 - beta) * log_fk) ** 2) + (1 / 1920) * (((1 - beta) * log_fk) ** 4))
            factor_3 = z / x
            factor_4 = 1 + ((1 - beta) ** 2 / 24 * (factor_1 ** 2) + (beta * nu * rho) / 4 * factor_1 + (
                    nu ** 2 * (2 - 3 * rho ** 2) / 24)) * time_ex
            model_vol = factor_1 * factor_2 * factor_3 * factor_4
        return model_vol
