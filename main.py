import os
import rsome as rso
import pandas as pd
import numpy as np
import warnings
from rsome import ro
from sklearn.decomposition import PCA as sklearnPCA
from typing import *

__author__ = 'kq'

HOME = os.environ['HOME'] + '/mdro/'
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)




def fetch_pc_returns(returns: pd.DataFrame,
                     components: Optional[int] = 5,
                     iterative: bool = False,
                     min_var: float = 0.9) -> pd.DataFrame:
    """

    :param returns:
    :param components:
    :param iterative:
    :param min_var:
    :return:
    """
    if iterative:
        pca = sklearnPCA(n_components=components)
        pc = pca.fit_transform(returns)
        varsum = sum(pca.explained_variance_ratio_)
        while varsum < min_var:
            components += 1
            pca = sklearnPCA(n_components=components)
            pca = pca.fit_transform(returns)
            varsum = sum(pca.explained_variance_ratio_)
    else:
        pca = sklearnPCA(n_components=components)

    pc = pca.fit_transform(returns)
    pcs = pca.components_
    pcs = pd.DataFrame(pcs, columns=returns.columns)
    pcs = pcs.div(pcs.sum(axis=1), axis=0)
    return pcs


def fetch_dro(returns: pd.DataFrame,
              epsilon: float = 0.01,  # Wasserstein radius
              num_components: int = 5,  # Number of factors
              rho: float = 10,  # Risk-aversion coefficient
              alpha: float = 0.2,  # Confidence (1-a)
              min_var: float = 0.9,  # Add component account until x% variance explained
              verbose: bool = True,  # Information print
              epsilon_list: Optional[List[float]] = []):
    """

    :param returns:
    :param epsilon:
    :param num_components:
    :param rho:
    :param alpha:
    :param min_var:
    :param verbose:
    :param epsilon_list:
    :return:
    """
    # Derived parameters
    n, m = returns.shape
    a1, b1 = -1, rho
    a2, b2 = -1 - rho / alpha, rho - rho / alpha
    C = - np.eye(m)
    d = np.ones(m)

    # Principal components of returns
    pcs = fetch_pc_returns(returns=returns,
                           components=num_components,
                           iterative=False,
                           min_var=min_var)

    pc_list = [(pcs.iloc[k] * returns) for k in list(pcs.index)]

    # MODEL
    model = ro.Model()
    x = model.dvar(m)
    lambdas = [model.dvar() for j in list(range(len(pc_list)))]
    omega = model.dvar()
    tau = model.dvar()
    gamma = model.dvar((2, n, m))
    model.st(gamma >= 0)
    tau = model.dvar()
    s_vars = [model.dvar(n) for j in list(range(len(pc_list)))]

    # Objective

    if not (len(epsilon_list) == len(pc_list)):
        epsilon_list = [epsilon] * len(pc_list)

    print(r'Using $\epsilon$ = {}$ for {} components'.format(epsilon_list, num_components))
    sumprod = sum([x * y for x, y in zip(epsilon_list, lambdas)])
    model.min(sumprod + ((1 / n) * sum([v.sum() for v in s_vars])))

    # Non-negative weights that sum to 1
    model.st(x.sum() == 1, x >= 0)

    # Factor constraints
    i = 0
    for pc in pc_list:
        yhat = np.array(pc)
        model.st(b1 * tau + a1 * (yhat @ x) + (gamma[0] * (1 + yhat)).sum(axis=1) <= s_vars[i])
        model.st(b2 * tau + a2 * (yhat @ x) + (gamma[1] * (1 + yhat)).sum(axis=1) <= s_vars[i])
        i += 1

    # Inf-norm constraint
    for j in range(len(pc_list)):
        for i in range(n):
            model.st(rso.norm(-gamma[0, i] - a1 * x, 'inf') <= lambdas[j])
            model.st(rso.norm(-gamma[1, i] - a2 * x, 'inf') <= lambdas[j])

            # Fetch solution
    if verbose:
        print(model.do_math())
    model.solve()
    if verbose:
        print(model.solution)
    weights = pd.Series(x.get(), index=returns.columns)
    return weights, pd.Series([j.get() for j in lambdas]), omega.get(), gamma.get(), tau.get()


def run(num_stocks: int, num_days: int, num_components: int) -> pd.Series:
    """

    :param num_stocks:
    :param num_days:
    :param num_components:
    :return:
    """
    data = pd.read_csv(HOME + '/spx_prices.csv', index_col=0, parse_dates=True).pct_change().iloc[1:].drop(
        ['BF.B', 'BRK.B'], axis=1)
    returns = data.iloc[-num_days:][data.columns[:num_stocks]]

    # Generate random epsilon
    epsilon_list = [(np.random.rand() / 100) for j in range(num_components)]
    weights, lambdas, omega, gamma, tau = fetch_dro(returns=returns,
                                                    num_components=num_components,
                                                    epsilon_list=epsilon_list)
    return weights
