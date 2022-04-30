import numpy as np
import toolkit as tk


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

        # E-step要用的量
        self.alpha = None  # 后面会重新赋值，应该是一个np.ndarray,列代表是时段，某列的第i行储存的就是alpha(z_{t, k}),k=0,...,K^nu
        self.beta = None


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
        self.Ob = np.hstack([self.O, o])

    def _forward(self):
        length = self.Ob.shape[0]
        # tentative: for developing
        self.emis = tk.multivariate_normal

        # work out alpha(z_1)
        alpha1 = self.pi * self.emis(self.Ob[0], self.phi)
        # start backward: warning: we may need to store all alpha's since incremental learning requires
        

    def _forward_init(self):
        cnt = np.arange(np.power(self.K, self.nu))
        cnt = np.floor(cnt / np.power(self.K, self.m))  # 处理后cnt的第i个元中的数字就是它对应的参数的序号



        