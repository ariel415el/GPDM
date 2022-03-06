import torch
from time import time
import numpy as np
import faiss
import pandas as pd


def get_NN_indices_faiss(X, Y, dim, index='flat', n_first=1):
    X = np.ascontiguousarray(X.cpu().numpy(), dtype='float32')
    Y = np.ascontiguousarray(Y.cpu().numpy(), dtype='float32')

    if index == 'flat':
        index = faiss.IndexFlat(dim)
    elif index == 'ivf':
        index = faiss.IndexIVFFlat(faiss.IndexFlat(dim), dim, 100)
        index.nprobe = 2
        index.train(Y)
    elif index == 'lsh':
        index = faiss.IndexLSH(dim)
    else:
        raise ValueError

    # index = faiss.index_cpu_to_all_gpus(index)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(Y)  # add vectors to the index

    _, I = index.search(X, n_first)  # actual search
    print(4)

    NNs = I[:, 0]

    if n_first == 1:
        return NNs
    else:
        return NNs, I


def efficient_compute_distances(X, Y):
    dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    d = X.shape[1]
    dist /= d  # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
    return dist


def get_col_mins_efficient(X, Y, b):
    mins = torch.zeros(Y.shape[0], dtype=X.dtype, device=X.device)
    n_batches = len(X) // b
    for i in range(n_batches):
        mins[i * b:(i + 1) * b] = efficient_compute_distances(X, Y[i * b:(i + 1) * b]).min(0)[0]
    if len(X) % b != 0:
        mins[n_batches * b:] = efficient_compute_distances(X, Y[n_batches * b:]).min(0)[0]

    return mins


def get_NN_indices_low_memory(X, Y, alpha, b=512):
    # Computes the distance of each y to the closest x and add alpha to get the per column normalizing factor
    normalizing_row = get_col_mins_efficient(X, Y, b=b)
    normalizing_row = alpha + normalizing_row[None, :]

    NNs = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    n_batches = len(X) // b
    for i in range(n_batches):
        dists = efficient_compute_distances(X[i * b:(i + 1) * b], Y) / normalizing_row
        NNs[i * b:(i + 1) * b] = dists.min(1)[1]
    if len(X) % b != 0:
        dists = efficient_compute_distances(X[n_batches * b:], Y) / normalizing_row
        NNs[n_batches * b:] = dists.min(1)[1]
    return NNs


class swd:
    def __init__(self, patch_size, n_proj=256):
        self.rand = torch.randn(3 * patch_size ** 2, n_proj).to(device)  # (slice_size**2*ch)
        self.__name__ = 'SWD'

    def __call__(self, X, Y):
        projx = torch.matmul(X, self.rand)
        projy = torch.matmul(Y, self.rand)

        projx, _ = torch.sort(projx, dim=0)
        projy, _ = torch.sort(projy, dim=0)

        loss = torch.abs(projx - projy).mean()


def time_call(func, X, Y, *args):
    func(X, Y, *args)
    times = []
    try:
        for i in range(n_reps):
            start = time()
            func(X, Y, *args)
            times.append(time()-start)
    except Exception as e:
        print(e)
        return None

    return np.mean(times), np.std(times)

def compute_ann_accuracy(s, p):
    n = s ** 2
    d = 3 * p ** 2
    X = torch.randn((n, d))
    Y = torch.randn((n, d))

    nn_fais, I = get_NN_indices_faiss(X, Y, d, 'ivf', n_first=10)
    nn = get_NN_indices_low_memory(X, Y, 1, b=256).numpy()

    recall = np.sum(nn_fais == nn) / nn.shape[0]
    n_recall = np.sum([nn[i] in I[i] for i in range(n)]) / nn.shape[0]
    dist_faiss = ((X - Y[nn_fais])**2).sum(1).mean()
    dist_true = ((X - Y[nn])**2).sum(1).mean()

    print(recall)
    print(n_recall)
    print(dist_faiss)
    print(dist_true)

    x =1

if __name__ == '__main__':
    # compute_ann_accuracy(128, 7)
    # exit()

    n_reps = 50
    device = torch.device('cuda:0')
    with torch.no_grad():
        full_report = pd.DataFrame()
        for s in [16]: # 64, 128, 256, 512, 1024]:
            for p in [7]:
                n = s ** 2
                d = 3 * p ** 2
                swd_loss = swd(p, n_proj=256)
                X = torch.randn((n, d)).to(device)
                Y = torch.randn((n, d)).to(device)

                print(s, p)
                # print(f"Pytorch-NN  ", time_call(get_NN_indices_low_memory, X, Y, 1, 256))
                # print(f"SWD         ", time_call(swd_loss, X, Y))
                print(f"faiss-PureNN", time_call(get_NN_indices_faiss, X, Y, d, 'flat'))
                # print(f"faiss-IVF   ", time_call(get_NN_indices_faiss, X, Y, d, 'ivf'))


