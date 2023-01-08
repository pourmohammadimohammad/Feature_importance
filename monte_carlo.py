import numpy as np

if __name__ == '__main__':
    d_ = 24 ** 2

    np.random.seed(2)
    theta = np.random.randn(1000000, d_)
    x1 = np.random.randn(1, d_)
    x2 = np.random.randn(1, d_)
    kernel = np.exp(- 0.5 * ((x1 - x2) ** 2).sum() / d_)
    approximation = list()
    for sample in np.arange(10, 1000000, 100):
        approximation += [(np.cos((x1 - x2) @ theta[:sample, :].T / np.sqrt(d_))).mean()]
    approximation = np.array(approximation)
