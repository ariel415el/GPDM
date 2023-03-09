import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

from tqdm import tqdm


class GMM1D:
    def __init__(self, n_components, data):
        """DATA shape is (n_hists, n_samples). Learn a separate 1d GMM on each hist samples"""
        super(GMM1D).__init__()
        self.n_components = n_components
        self.pi = []
        self.mu = []
        self.sigma = []
        print("Fitting GMMS...", end='...')
        for i in tqdm(range(len(data))):
            gmm = GaussianMixture(n_components=n_components, covariance_type='diag') #verbose=2, verbose_interval=1)
            gmm.fit(data[i].unsqueeze(1).cpu().numpy())
            self.pi += [torch.from_numpy(gmm.weights_).to(data.device)]
            self.mu += [torch.from_numpy(gmm.means_[..., 0]).to(data.device)]
            self.sigma += [torch.sqrt(torch.from_numpy(gmm.covariances_[..., 0])).to(data.device)]

        print("Done")
        self.pi = torch.stack(self.pi)
        self.mu = torch.stack(self.mu)
        self.sigma = torch.stack(self.sigma)

    def sample(self, n):
        indices = torch.multinomial(self.pi, n, replacement=True)
        # samples = self.mu[:, indices] + self.sigma[indices] * torch.randn(n)[None, :, None, None]
        mus = torch.stack([self.mu[i, indices[i]] for i in range(len(self.mu))])
        sigmas = torch.stack([self.sigma[i, indices[i]] for i in range(len(self.sigma))])
        rands = torch.randn(len(self.mu), n).to(self.mu.device)
        samples = mus + sigmas * rands
        return samples


class GMMnD:
    def __init__(self, n_components, data):
        """DATA shape is (n_hists, n_samples). Learn a separate 1d GMM on each hist samples"""
        super(GMM1D).__init__()
        self.n_components = n_components
        print("Fitting GMMS...", end='...')
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag')  # verbose=2, verbose_interval=1)
        gmm.fit(data.permute(1,0).cpu().numpy()) # n_samples of dim n_hists
        self.pi = torch.from_numpy(gmm.weights_).to(data.device)
        self.mu = torch.from_numpy(gmm.means_).to(data.device)
        self.sigma = torch.sqrt(torch.from_numpy(gmm.covariances_)).to(data.device)

    def sample(self, n):
        indices = torch.multinomial(self.pi, n, replacement=True)
        samples = self.mu[indices] + self.sigma[indices] * torch.randn(n)[:, None].to(self.mu.device)
        return samples.T


if __name__ == '__main__':
    n_samples = 100000
    b = 3
    x1 = np.random.randn(b, n_samples // 2)
    x2 = np.random.randn(b, n_samples // 2)

    for i in range(b):
        x1[i] = x1[i] * np.random.randn()*0.1 + np.random.randn()
        x2[i] = x2[i] * np.random.randn()*0.1 + np.random.randn()

    x = np.concatenate([x1,x2], axis=1)

    # gmm = GMM1D(2, torch.from_numpy(x))
    gmm = GMMnD(2, torch.from_numpy(x))
    y = gmm.sample(n_samples)

    for i in range(b):
        plt.hist([x[i, :],y[i, :]],label=['x', 'y'], bins=100)
        plt.legend()
        plt.show()
        plt.clf()

