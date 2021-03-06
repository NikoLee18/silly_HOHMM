import numpy as np
from . import toolkit as tk
from tqdm import tqdm


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
        self.mini = min(r, m)
        self.numEm = np.power(self.K, m)   # 提前计算好可以有多少个emission matrix
        self.max_perd = max_perd
        self.emis = tk.multivariate_normal      # 在这里传入正态分布的密度函数

        self.Ob = None

        # E-step要初始化的量
        self.alpha = None  # 后面会重新赋值，应该是一个np.ndarray,列代表是时段，某列的第i行储存的就是alpha(z_{t, k}),k=0,...,K^nu
        self.beta = None
        # E-step计算的东西   ！ update !这个不能在这里初始化就不管了，每次都更新了self.phi的！
        self.gauss_dist = tk.NormalDist(self.phi)   # initialization (based on pi)
        index_func = tk.index_transfer(self.K, self.r, self.m)
        self.indexes = index_func(np.arange(np.power(self.K, self.nu)))
        self.gamma, self.xi = None, None
        # scaling factor
        self.sf = None

        # 记录训练过程中p(X)的上升过程
        self.pX = None



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

    def learn(self, num_epoch, surveillance=False):
        if surveillance:
            self.pX = []
            for _ in tqdm(range(num_epoch)):
                self._run_one_cycle()
                print(self.phi[0], "\n")
                print(self.A, "\n")
                print(self.pi, "\n")
                pX = np.log(np.array(self.sf)).sum() + np.log(self.c1)
                print(pX,"\n")
                self.pX.append(pX)
            return self.pX
        else:
            for _ in tqdm(range(num_epoch)):
                self._run_one_cycle()
            return None


    def _run_one_cycle(self):
        """在参数初始化好了之后（相当于M-步过了）， 跑一个E-M过程"""
        self.E_step()
        self.M_step()

    def M_step(self):
        """需要更新的几个参数，M-step都已经在self._pi_hat, self._A_hat, self._mu_sigma_hat 中写好了，调用一下就行"""
        self._pi_hat()
        self._A_hat()
        self._mu_sigma_hat()
        # 更新完就可以了，进入下一个E-step根据更新过的东西去算充分统计量
        self.gauss_dist = tk.NormalDist(self.phi)   # 更新一下分布密度的函数！

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
        gmsum = self.gamma.sum(axis=0).reshape(np.power(self.K, self.m), -1).sum(axis=1).reshape(-1, 1)
        # Gamma Sum用来当两个的分母,这里要reshape成一列，因为我们接下来用i行来表示第i中扩散参数
        # 现在开始算mu的denom
        gm = self.gamma.reshape(self.gamma.shape[0], np.power(self.K, self.m), -1).sum(axis=2).T  # 把gamma弄成横着长条形状
        res_mu = gm @ self.Ob / gmsum # 有K^m行，列数和Observe一样多。 每一行代表一个mu_i的估计
        # 现在算sigma的估计
        mu = np.stack([i[0] for i in self.phi], axis=0)
        centrl_obs = self.Ob - mu.reshape((mu.shape[0], -1, mu.shape[1])) # 这里用了一下broadcast机制，axis0表示参数种类
        # axis1表示时间步，axis2是mu/观测值的维数
        gm = np.stack([gm] * self.Ob.shape[1], axis=2)   # 这一步还是在调整分子上最后一部分的gamma，公式3.28里三个相乘没办法只用
        # 矩阵乘法，所以我们把一部分和中心化后的观测阵做piecewise multiplication,然后再矩阵乘法
        A = centrl_obs.transpose(0, 2, 1) @ (centrl_obs * gm)       # 这样就得到了一个离差阵（张量），axis0还是表示参数种类
        res_sigma = A / gmsum.reshape(-1, 1, 1)    # 在axis0上，分别除以相应的公式3.28里的分母上的量
        # 现在把mu, Sigma更新回class attributes里
        self.phi = [(res_mu[i], res_sigma[i]) for i in range(res_mu.shape[0])]
        return self.phi



    def E_step(self):
        alpha = self._forward()
        beta = self._backward()
        # 下面算两个充分统计量 gamma, xi。不过首先算p(\textbf X)
        self.alpha = np.vstack(alpha)    # 整理好alpha，vstack成 N * K^nu
        self.beta = np.hstack(beta).T    # 整理好beta，先hstack成 K^nu * N的，再转置一下，和alpha形状一样。保持数据风格，用行代表时间步
        gamma = self.alpha * self.beta    # 按元素相乘再除以一个p(X)，得到全部情况的gamma充分统计量。 纵向是时间步，横向是q_n取不同值

        xi = []
        for t in range(self.Ob.shape[0]-1):
            xi_n = (alpha[t] * self.emission_dens[t]).reshape(-1,1) * self.A * beta[t+1]
            xi.append(xi_n)
        xi = np.stack(xi, axis=0)
        # 数值稳定版本的xi算法和原来不太一样了
        array_scaling_factor = np.array(self.sf).reshape(-1, 1, 1)
        xi = xi * array_scaling_factor
        # 存入class attribute
        self.gamma = gamma
        self.xi = xi
        return gamma, xi

    def _forward(self):
        alpha = [self._forward_init()]    # 初始化alpha
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
        print("emis: ", _emis)
        c1 = _emis @ self.pi
        self.c1 = c1
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








