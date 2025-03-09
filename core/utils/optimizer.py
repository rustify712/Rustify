import numpy as np


class TemperatureOptimizer:
    """温度优化器
    用于采样温度值，以及根据当前的温度和评分，调整当前最佳温度。

    Args:
        initial_temp (float): 初始温度
        min_temp (float): 最小温度
        max_temp (float): 最大温度
        step (float): 温度调整步长
        sigma (float): 正态分布的标准差，用于采样
    """

    def __init__(self, initial_temp=0.4, min_temp=0.0, max_temp=1.3, step=0.01, sigma=0.1):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.sigma = sigma
        self.step = step
        self.current_temp = initial_temp

    def do_sample(self, num_samples: int = 1) -> list[float]:
        """采样
        根据当前最佳温度，从正态分布中采样 num_samples 个温度值。
        """
        samples = []
        while len(samples) < num_samples:
            sample = np.random.normal(loc=self.current_temp, scale=self.sigma)
            if sample in samples:
                continue
            if sample < self.min_temp or sample > self.max_temp:
                continue
            samples.append(round(sample, 2))
        return samples

    def update(self, temperature: float):
        """更新
        根据当前的温度和评分，调整当前最佳温度。
        """
        if temperature < self.current_temp:
            self.current_temp -= self.step
        elif temperature > self.current_temp:
            self.current_temp += self.step
        if self.current_temp < self.min_temp:
            self.current_temp = self.min_temp
        if self.current_temp > self.max_temp:
            self.current_temp = self.max_temp
        return self.current_temp
