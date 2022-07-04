import numpy as np
import torch
from scipy import linalg
import torch.nn as nn
from torchvision import models

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


class InceptionV3FeatureExtractor(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    def __init__(self, n_convblocks=4):
        super(InceptionV3FeatureExtractor, self).__init__()

        self.blocks = nn.ModuleList()
        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if n_convblocks >= 2:
            block1 = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if n_convblocks >= 3:
            block2 = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if n_convblocks >= 4:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
            ]
            self.blocks.append(nn.Sequential(*block3))

        if n_convblocks >= 5:
            block4 = [
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block4))

    def extract_featuremaps(self, inp):
        """Get Inception feature maps"""
        x = 2 * inp - 1  # Scale from range (0, 1) to range (-1, 1)
        for idx, block in enumerate(self.blocks):
            x = block(x)
        return x

    def get_statistics(self, batch):
        pred = self.extract_featuremaps(batch / 255)
        b, c, h, w = pred.shape
        assert b == 1
        act = pred.cpu().numpy().transpose(0, 2, 3, 1).reshape(h*w, c)

        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)

        return mu, sigma


def compute_SIFID(refernce_images, generated_images, n_convblocks=4, device=torch.device("cuda:0")):
    """
    :param refernce_images: (1, c, h, w)
    :param generated_images: (b, c, h, w)
    :return:
    """
    with torch.no_grad():
        model = InceptionV3FeatureExtractor(n_convblocks).to(device)
        ref_mu, ref_sigma = model.get_statistics(refernce_images.to(device))
        sfid_scores = []
        for i in range(generated_images.shape[0]):
            mu, sigma = model.get_statistics(generated_images[i].unsqueeze(0).to(device))
            sifid = calculate_frechet_distance(ref_mu, ref_sigma, mu, sigma)
            sfid_scores.append(sifid)

    return np.mean(sfid_scores)
