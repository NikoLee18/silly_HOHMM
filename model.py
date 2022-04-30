import numpy as np
from . import toolkit as tk


class SillyHMM:
    """
    A: Init transition matrix
    phi: Init params for emission
    pi: Init prob. of first state (or more if order > 1)
    numS: Number of states. Make sure it is greater than 1
    order: Order of this Hidden Markov Model
    emis: Emission prob. function, it should take two params: first one z, second one emission param



    """

    def __init__(self, A, phi, pi, num_states:int, r:int, m:int, max_perd:int):
        """
        Init a new Hidden Markov Model with any order
        :param A: Init transition matrix
        :param phi: Init params for emission
        :param pi: Init prob. of first state (or more if order > 1)
        :param num_states: Number of states. Make sure it is greater than 1

        :param max_perd: max period to be included in training. If recursively fed data exceeds it,
        we will drop first few observations to keep required length.
        """
        self.A = A
        self.phi = phi
        self.pi = pi
        # 和模型的阶数有关的东西
        assert num_states > 1
        self.K = num_states   # 隐藏状态的个数，这里用K，也就是没有经过变换的原始状态的K
        self.r = r            # transit的阶数
        self.m = m            # emission的阶数
        self.nu = max(r, m)
        self.numEm = np.power(self.K, m)   # 提前计算好可以有多少个emission matrix
        self.max_perd = max_perd
        self.emis = tk.multivariate_normal      # 在这里传入正态分布的密度函数

        self.Ob = None

        # E-step要初始化的量
        self.alpha = None  # 后面会重新赋值，应该是一个np.ndarray,列代表是时段，某列的第i行储存的就是alpha(z_{t, k}),k=0,...,K^nu
        self.beta = None
        # E-step计算的东西
        self.gauss_dist = tk.NormalDist(self.phi)   # initialization (based on pi)
        index_func = tk.index_transfer(self.K, self.r, self.m)
        self.indexes = index_func(np.arange(np.power(self.K, self.nu)))
        self.gamma, self.xi = None, None



    def feed_batch_data(self, O):
        """
        Feed pre training data
        :param O: Observations
        :return:
        """
        self.Ob = O

    def feed_one(self, o):
        """
        Feed additional one data point
        :param o:
        :return:
        """
        self.Ob = np.vstack([self.Ob, o])

    def M_step(self):
        pass

    def _pi_hat(self):
        s = self.gamma[0]
        self.pi = s / s.sum()
        return self.pi

    def _A_hat(self):
        # 论文上的公式(4.3)，我们分分子分母算，然后用一个piecewise矩阵除法来得到A
        numerator = self.xi.sum(axis=0)    # 分子挺好算的，直接按照时间轴压在一起求和就好了，得到的是 2d array
        denom = self.xi.sum(axis=0).sum(axis=1)  # 先按照时间求和，再按照xi(q_n-1,q_n)的q_n求和，得到1d array
        denom = np.vstack([denom] * np.power(self.K, self.nu)).T

        self.A = numerator / denom
        return self.A

    def _mu_sigma_hat(self):
        # 先算两个重要的量，参考论文公式(4.4),(4.5)，不过用矩阵整理好的就挺好整
        Gm = self.gamma.sum(axis=0)


    def E_step(self):
        alpha = self._forward()
        beta = self._backward()
        # 下面算两个充分统计量 gamma, xi。不过首先算p(\textbf X)
        self.alpha = np.vstack(alpha)    # 整理好alpha，vstack成 N * K^nu
        self.beta = np.hstack(beta).T    # 整理好beta，先hstack成 K^nu * N的，再转置一下，和alpha形状一样。保持数据风格，用行代表时间步
        pX = self.alpha[-1].sum()
        gamma = self.alpha * self.beta / pX   # 按元素相乘再除以一个p(X)，得到全部情况的gamma充分统计量。 纵向是时间步，横向是q_n取不同值

        xi = []
        for t in range(self.Ob.shape[0]-1):
            xi_n = (alpha[t] * self.emission_dens[t]).reshape(-1,1) * self.A * beta[t+1]
            xi.append(xi_n)
        xi = np.stack(xi, axis=0)
        return gamma, xi


    def _forward(self):
        alpha = [self._forward_init()] # 初始化alpha
        _factor = np.power(self.K, self.nu - self.r)

        # update: 分布不用算很多遍，由于先运行forward，直接存起来，会多次使用
        arr_tmp = np.hstack([np.vstack([self.Ob[0]] * len(self.indexes)), self.indexes.reshape(-1, 1)])
        _emis = np.apply_along_axis(self.gauss_dist, axis=1, arr=arr_tmp)  # 得到扩散概率的1*K^nu的矩阵
        self.emission_dens = [_emis]

        # 一个E步骤的epoch
        for t in range(1, self.Ob.shape[0]):

            arr_tmp = np.hstack([np.vstack([self.Ob[t]] * len(self.indexes)), self.indexes.reshape(-1, 1)])
            _emis = np.apply_along_axis(self.gauss_dist, axis=1, arr=arr_tmp) # 得到扩散概率的1*K^nu的矩阵
            self.emission_dens.append(_emis)
            tmp = alpha[t-1] @ self.A    # 计算后面两个，然后再做kroneker乘积
            alpha.append(_emis.reshape(1, -1)[0] * tmp / _factor)
        return alpha

    def _forward_init(self):
        """
        专供_forward方法调用，用于计算初始的alpha_1
        :return:
        """

        # temp_vars   在真正代码中改掉！！！！！
        # tmp = np.zeros((len(indexes), 3))

        arr_tmp = np.hstack([np.vstack([self.Ob[0]] * len(self.indexes)), self.indexes.reshape(-1, 1)])
        alpha1 = np.apply_along_axis(self.gauss_dist, axis=1, arr=arr_tmp)
        return alpha1

    def _backward(self):
        # backward的起始比较简单，初始化一个全是1的就行
        dim = np.power(self.K, self.nu)
        _factor = np.power(self.K, self.nu - self.r)

        # 开始向后算，不过储存还是向前存
        beta = [np.ones((dim, 1))]
        for t in range(1, self.Ob.shape[0]):
            leng = self.Ob.shape[0]
            _emis = self.emission_dens[leng - t].reshape(-1, 1)  # 记得换成一列的形式

            neo_beta = self.A @ (_emis * beta[t-1]) / _factor  # 计算新的一个beta，在算法上是要放到上一个的“后面”，但是存在“前面”
            beta.append(neo_beta)
        return list(reversed(beta))  # 最后记得给它翻回来


class HMM:
    """
    A: Init transition matrix
    phi: Init params for emission
    pi: Init prob. of first state (or more if order > 1)
    numS: Number of states. Make sure it is greater than 1
    order: Order of this Hidden Markov Model
    emis: Emission prob. function, it should take two params: first one z, second one emission param



    """

    def __init__(self, A, phi, pi, num_states:int, r:int, m:int, max_perd:int):
        """
        Init a new Hidden Markov Model with any order
        :param A: Init transition matrix
        :param phi: Init params for emission
        :param pi: Init prob. of first state (or more if order > 1)
        :param num_states: Number of states. Make sure it is greater than 1

        :param max_perd: max period to be included in training. If recursively fed data exceeds it,
        we will drop first few observations to keep required length.
        """
        self.A = A
        self.phi = phi
        self.pi = pi
        # 和模型的阶数有关的东西
        assert num_states > 1
        self.K = num_states   # 隐藏状态的个数，这里用K，也就是没有经过变换的原始状态的K
        self.r = r            # transit的阶数
        self.m = m            # emission的阶数
        self.nu = max(r, m)
        self.numEm = np.power(self.K, m)   # 提前计算好可以有多少个emission matrix
        self.max_perd = max_perd
        self.emis = tk.multivariate_normal      # 在这里传入正态分布的密度函数

        self.Ob = None

        # E-step要初始化的量
        self.alpha = None  # 后面会重新赋值，应该是一个np.ndarray,列代表是时段，某列的第i行储存的就是alpha(z_{t, k}),k=0,...,K^nu
        self.beta = None
        # E-step计算的东西
        self.gauss_dist = tk.NormalDist(self.phi)   # initialization (based on pi)
        index_func = tk.index_transfer(self.K, self.r, self.m)
        self.indexes = index_func(np.arange(np.power(self.K, self.nu)))
        self.gamma, self.xi = None, None
        # scaling factor
        self.sf = None



    def feed_batch_data(self, O):
        """
        Feed pre training data
        :param O: Observations
        :return:
        """
        self.Ob = O

    def feed_one(self, o):
        """
        Feed additional one data point
        :param o:
        :return:
        """
        self.Ob = np.vstack([self.Ob, o])

    def M_step(self):
        pass

    def _pi_hat(self):
        s = self.gamma[0]
        self.pi = s / s.sum()
        return self.pi

    def _A_hat(self):
        # 论文上的公式(4.3)，我们分分子分母算，然后用一个piecewise矩阵除法来得到A
        numerator = self.xi.sum(axis=0)    # 分子挺好算的，直接按照时间轴压在一起求和就好了，得到的是 2d array
        denom = self.xi.sum(axis=0).sum(axis=1)  # 先按照时间求和，再按照xi(q_n-1,q_n)的q_n求和，得到1d array
        denom = np.vstack([denom] * np.power(self.K, self.nu)).T

        self.A = numerator / denom
        return self.A

    def _mu_sigma_hat(self):
        # 先算两个重要的量，参考论文公式(4.4),(4.5)，不过用矩阵整理好的就挺好整
        Gm = self.gamma.sum(axis=0)


    def E_step(self):
        alpha = self._forward()
        beta = self._backward()
        # 下面算两个充分统计量 gamma, xi。不过首先算p(\textbf X)
        self.alpha = np.vstack(alpha)    # 整理好alpha，vstack成 N * K^nu
        self.beta = np.hstack(beta).T    # 整理好beta，先hstack成 K^nu * N的，再转置一下，和alpha形状一样。保持数据风格，用行代表时间步
        pX = self.alpha[-1].sum()
        gamma = self.alpha * self.beta / pX   # 按元素相乘再除以一个p(X)，得到全部情况的gamma充分统计量。 纵向是时间步，横向是q_n取不同值

        xi = []
        for t in range(self.Ob.shape[0]-1):
            xi_n = (alpha[t] * self.emission_dens[t]).reshape(-1,1) * self.A * beta[t+1]
            xi.append(xi_n)
        xi = np.stack(xi, axis=0)
        # 数值稳定版本的xi算法和原来不太一样了
        array_scaling_factor = np.array(self.sf, ndmin=3)
        xi = xi / array_scaling_factor

        return gamma, xi


    def _forward(self):
        alpha = [self._forward_init()] # 初始化alpha
        _factor = np.power(self.K, self.nu - self.r)

        # update: 分布不用算很多遍，由于先运行forward，直接存起来，会多次使用
        arr_tmp = np.hstack([np.vstack([self.Ob[0]] * len(self.indexes)), self.indexes.reshape(-1, 1)])
        _emis = np.apply_along_axis(self.gauss_dist, axis=1, arr=arr_tmp)  # 得到扩散概率的1*K^nu的矩阵
        self.emission_dens = [_emis]
        self.sf = list()
        # 一个E步骤的epoch
        for t in range(1, self.Ob.shape[0]):
            arr_tmp = np.hstack([np.vstack([self.Ob[t]] * len(self.indexes)), self.indexes.reshape(-1, 1)])
            _emis = np.apply_along_axis(self.gauss_dist, axis=1, arr=arr_tmp) # 得到扩散概率的1*K^nu的矩阵
            self.emission_dens.append(_emis)
            tmp = alpha[t-1] @ self.A    # 计算后面两个，然后再做kroneker乘积
            # 计算scaling factor
            c = self._scaling_factor(_emis.reshape(1, -1), self.A, alpha[t-1].reshape(-1, 1), _factor)
            self.sf.append(c)

            alpha.append(_emis.reshape(1, -1)[0] * tmp / c)  # 这里加上除以scaling factor
        return alpha

    def _forward_init(self):
        """
        专供_forward方法调用，用于计算初始的alpha_1
        :return:
        """

        # temp_vars   在真正代码中改掉！！！！！
        # tmp = np.zeros((len(indexes), 3))

        arr_tmp = np.hstack([np.vstack([self.Ob[0]] * len(self.indexes)), self.indexes.reshape(-1, 1)])
        alpha1 = np.apply_along_axis(self.gauss_dist, axis=1, arr=arr_tmp)

        # 计算scaling factor
        _emis = np.apply_along_axis(self.gauss_dist, axis=1, arr=arr_tmp)  # 得到扩散概率的1*K^nu的矩阵
        c1 = _emis @ self.pi
        return alpha1 / c1

    def _backward(self):
        # backward的起始比较简单，初始化一个全是1的就行
        dim = np.power(self.K, self.nu)
        # _factor = np.power(self.K, self.nu - self.r)

        # 开始向后算，不过储存还是向前存
        beta = [np.ones((dim, 1))]
        for t in range(1, self.Ob.shape[0]):
            leng = self.Ob.shape[0]
            _emis = self.emission_dens[leng - t].reshape(-1, 1)  # 记得换成一列的形式

            neo_beta = self.A @ (_emis * beta[t-1])  # 计算新的一个beta，在算法上是要放到上一个的“后面”，但是存在“前面”
            beta.append(neo_beta / self.sf[leng - t - 1])  # 这里也除以scaling factor
        return list(reversed(beta))  # 最后记得给它翻回来

    @staticmethod
    def _scaling_factor(emis: np.ndarray, A: np.ndarray, alpha:np.ndarray, f):
        """

        :param emis:
        :param A:
        :param alpha:
        :param f:
        :return:
        """
        assert emis.shape[0] == 1 and A.shape[0] == emis.shape[1] and A.shape[1] == alpha.shape[0]
        return (emis @ A @ alpha / f)[0][0]








