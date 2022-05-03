import numpy as np


def init_multinomial(dim: tuple, method='uniform'):
    """

    :param dim: tuple; if you want to initiate transition matrix, input (S^nu, S^nu)
    if want to init prior distribution matrix, use: (S^nu,).  But remember to input a tuple
    :param method: 'uniform' or some others I may update
    :return:
    """
    if method == 'uniform':
        return np.ones(dim)/dim[0]
    else:
        raise


def init_gaussian_emission(K, m, dim, mean=0, scale=0.01):
    re = []
    for _ in range(np.power(K, m)):
        re.append(_init_gaussian_emission(dim, mean, scale))
    return re

def _init_gaussian_emission(dim: int, mean=0, scale=0.01):
    if type(dim) is int or np.intc:
        return _init_one_dim(dim, mean, scale)
    else:
        raise Exception("请检查输入的维度，应该是int")


def init_hidden_state_dist(dim: int, scale=0.1):
    if type(dim) is int or np.intc:
        return _init_one_dim(dim, 0, scale)[0]
    else:
        raise Exception("请检查输入的维度，应该是int")


class NormalDist:
    def __init__(self, phi):
        self.phi = phi

    def __call__(self, arg):
        return multivariate_normal(arg[:-1], self.phi[int(arg[-1])])


def multivariate_normal(x: np.ndarray, phi:tuple):
    """

    :param x: 1 dim array
    :param phi: 1. mu 2. sigma
    :return:
    """
    dim = len(x)
    # sigma_inv * (x - mu) can be obtained by solving sigma * X = x - mu and this is rather simpler
    sigma_inv_x = np.linalg.solve(phi[1], x - phi[0])
    # sufficient statistics
    suff_stat = -(x - phi[0]).T @ sigma_inv_x / 2
    density = np.power(2 * np.pi, -dim / 2) * np.power(np.linalg.det(phi[1]), -dim) * \
              np.exp(suff_stat)
    return density


def init_transit_matrix(K, r, m):
    nu = max(r, m)
    dim = np.power(K, nu)
    feasible = feasible_transit(K, nu)
    A = np.zeros((dim, dim))
    A[feasible[0], feasible[1]] = np.random.randn(dim * K)
    A = np.where(A > 0, A, -A)
    A = A / (np.array([1] * dim) @ A)
    return A


def redundancies(K:int, r:int, m:int):
    """
    根据Hardar Messer的方法来算出有冗余的转移/扩散的参数。 也就是计算I_r(K)和I_m(K)以及可行的转移概率
    :param K: number of hidden states(不是转换过的K^\nu)
    :param r: 状态转移的阶数
    :param m: 扩散概率的阶数
    :return: 三个I_r, I_m, E
    """
    return _gen_redund_I(K, r, m), feasible_transit(K, max(r, m))


def feasible_transit(K: int, nu: int):
    """
    a transitio is feasible if and only if:
    i_{nu-1} + \sum_{l=0}^{\nu-2} i_l K^{\nu - 1 - l}  -->  j_0 + \sum_{l=0}^{\nu - 2} i_l K^{\nu - 2 - l}
    :param K:
    :param nu:
    :return: 两个np.ndarray, 只要一个(i, j)对满足：i 属于 former, j 属于 latter就可以转移
    """
    feas = np.arange(0, np.power(K, nu - 1))  # feas储存可转移情形中两边相同的部分，也即共享的nu-1个状态
    former = np.arange(0, K).reshape(-1, 1)  # 做成一列，方便利用numpy进行自动扩展，加和
    former = (feas * K + former).reshape(1, -1)[0]
    former = np.vstack([former] * K).reshape(1, -1)[0]

    latter = np.arange(0, K).reshape(-1, 1)
    latter = np.hstack([latter] * K).reshape(-1, 1) * np.power(K, nu - 1)
    latter = (feas + latter).reshape(1, -1)[0]
    return former, latter


def index_transfer(K, r, m):
    """用来生成一个判断带冗余表示的index对应另一个的index"""
    mini = min(r, m)
    nu = max(r, m)

    def temp(index):
        index = np.array(index)   # 无论index是一个array还是一个int，都给他换成array
        if False in (index < (np.power(K, nu) - 0.9) * np.ones_like(index)):
            raise Exception("输入的index超出范围了")
        elif m >= r:
            return index
        else:
            return np.floor(index / np.power(K, nu - mini))
    return temp


def _init_one_dim(dim, mean=0, scale=0.1):
    """

    :param dim: dim of observation variables
    :param scale: stdDev of mu
    :return: initiated mu and sigma. sigma is SPD matrix and mu is a multivariate gaussian
    """
    mu = np.random.randn(dim) * scale + mean
    N = dim * 3
    tmp = np.random.randn(dim, N) * scale + mean
    sigma = tmp @ tmp.T / N
    return mu, sigma


def _gen_redund_I(K:int, r: int, m: int):
    """
    产生一个可以返回tuple(I_r(i), I_m(i))的函数.
    :param K: 隐藏状态数
    :param r: 状态转移的阶数
    :param m: 扩散概率的阶数
    :return: 一个函数，func(i)返回tuple(I_r(i), I_m(i))
    """
    nu = max(r, m)
    # denom = np.power(K, nu)
    # r, m = max(r, m), min(r, m)
    def gen(i):
        return _gen_I_r(i, K, nu-r), _gen_I_r(i, K, nu-m)
    return gen


def _gen_I_r(i, K, nuk):
    denom = np.power(K, nuk)
    tmp = np.floor(i / denom)*denom
    return np.arange(tmp, tmp+denom)

