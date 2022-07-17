import cv2
import torch
from time import time
import numpy as np
import faiss
import pandas as pd

"""
NN search modules taken from https://github.com/ariel415el/Efficient-GPNN
"""


class FaissFlat:
    def __init__(self, n_first=1, use_gpu=False):
        self.use_gpu = use_gpu
        self.n_first = n_first
        self.name = "FaissFlat" + ("GPU" if self.use_gpu else "CPU") + ")"

    def init_index(self, Y):
        dim = Y.shape[-1]
        self.index = faiss.IndexFlatL2(dim)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def __call__(self, X, Y):
        X = np.ascontiguousarray(X.numpy(), dtype='float32')
        Y = np.ascontiguousarray(Y.numpy(), dtype='float32')
        self.init_index(Y)
        self.index.add(Y)  # add vectors to the index

        _, I = self.index.search(X, self.n_first)  # actual search

        NNs = I[:, 0]

        if self.n_first == 1:
            return NNs
        else:
            return NNs, I


class FaissIVF(FaissFlat):
    def __init__(self, nprobe=1, n_first=1, use_gpu=False):
        super(FaissIVF, self).__init__(n_first, use_gpu)
        self.nprobe = nprobe
        self.name = f"FaissIVF(nprobe={nprobe}, " + ("GPU" if self.use_gpu else "CPU") + ")"

    def _get_index(self, n, d):
        return faiss.IndexIVFFlat(faiss.IndexFlat(d), d, int(np.sqrt(n)))

    def init_index(self, Y):
        n, d = Y.shape

        self.index = self._get_index(n, d)

        self.index.nprobe = self.nprobe

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.train(Y)


class FaissIVFPQ(FaissIVF):
    def __init__(self, nprobe=1, n_first=1, use_gpu=False):
        super(FaissIVFPQ, self).__init__(nprobe, n_first, use_gpu)
        self.name = f"FaissIVF-PQ(nprobe={nprobe}, " + ("GPU" if self.use_gpu else "CPU") + ")"

    def _get_index(self, n, d):
        return faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, int(np.sqrt(n)), 8, 8)


class PytorchNN:
    def __init__(self, batch_size=256, alpha=1, use_gpu=False):
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = torch.device("cuda:0" if use_gpu else 'cpu')
        self.name = "PytorchNN(" + ("GPU" if use_gpu else "CPU") + ")"

    def __call__(self, X, Y):
        return PytorchNN.get_NN_indices_low_memory(X.to(self.device), Y.to(self.device), self.alpha, self.batch_size).cpu().numpy()
        
    @staticmethod
    def efficient_compute_distances(X, Y):
        dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
        d = X.shape[1]
        dist /= d  # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
        return dist

    @staticmethod
    def get_col_mins_efficient(X, Y, b):
        mins = torch.zeros(Y.shape[0], dtype=X.dtype, device=X.device)
        n_batches = len(X) // b
        for i in range(n_batches):
            mins[i * b:(i + 1) * b] = PytorchNN.efficient_compute_distances(X, Y[i * b:(i + 1) * b]).min(0)[0]
        if len(X) % b != 0:
            mins[n_batches * b:] = PytorchNN.efficient_compute_distances(X, Y[n_batches * b:]).min(0)[0]

        return mins

    @staticmethod
    def get_NN_indices_low_memory(X, Y, alpha, b=512):

        NNs = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        n_batches = len(X) // b
        for i in range(n_batches):
            dists = PytorchNN.efficient_compute_distances(X[i * b:(i + 1) * b], Y)
            NNs[i * b:(i + 1) * b] = dists.min(1)[1]
        if len(X) % b != 0:
            dists = PytorchNN.efficient_compute_distances(X[n_batches * b:], Y)
            NNs[n_batches * b:] = dists.min(1)[1]
        return NNs


class swd:
    def __init__(self, patch_size=7, n_proj=256, use_gpu=False):
        self.device = torch.device("cuda:0" if use_gpu else 'cpu')
        self.name = f"SWD(n_proj={n_proj}, " + ("GPU" if use_gpu else "CPU") + ")"
        self.rand = torch.randn(3 * patch_size ** 2, n_proj).to(self.device)  # (slice_size**2*ch)

    def __call__(self, X, Y):
        projx = torch.matmul(X.to(self.device), self.rand)
        projy = torch.matmul(Y.to(self.device), self.rand)

        projx, _ = torch.sort(projx, dim=0)
        projy, _ = torch.sort(projy, dim=0)

        loss = torch.abs(projx - projy).mean().cpu()
        
        return loss


def measure_funtion_call_time(func, X, Y, *args):
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


def get_vectors_from_img(path):
    img = cv2.imread(path)[None, :]
    unfold = torch.nn.Unfold(kernel_size=p, stride=1)
    vecs = unfold(torch.from_numpy(img).float().permute(0,3,1,2))[0].T
    return vecs


def test_ann_accuracy(n=10):
    """Compute the accuracy of various aproximate Faiss nearest neighbor methods compared with exact NN on real images"""
    # exact_NN = FaissFlat(use_gpu=True, n_first=1)
    exact_NN = PytorchNN(use_gpu=True)
    Aproximagte_NNs = [
        FaissIVF(nprobe=1, n_first=n, use_gpu=True),
        # FaissIVF(nprobe=10, n_first=10, use_gpu=True),
        FaissIVFPQ(nprobe=1, n_first=n, use_gpu=True),
        # FaissIVFPQ(nprobe=10, n_first=10, use_gpu=True),
    ]

    recall_1s = []
    recall_ns = []
    dist_overheads_false= []

    for ANN in Aproximagte_NNs:
        table = pd.DataFrame()
        for i in range(1, 17):

            X = get_vectors_from_img(f'../images/SIGD16/{i}.jpg')

            # Split image patches into two random sets
            choice = np.random.choice(range(X.shape[0]), size=(X.shape[0]//2,), replace=False)
            ind = np.zeros(X.shape[0], dtype=bool)
            ind[choice] = True
            Y = X[ind]
            X = X[~ind]

            n, d = X.shape
            exact_nn = exact_NN(X, Y)

            ann, I = ANN(X, Y)

            recall = np.sum(ann == exact_nn) / exact_nn.shape[0]
            recall_n = np.sum([exact_nn[i] in I[i] for i in range(n)]) / exact_nn.shape[0]

            nn_dists = ((X - Y[exact_nn])**2)[ann != exact_nn].sum(1).mean().item()
            ann_dists = ((X - Y[ann])**2)[ann != exact_nn].sum(1).mean().item()
            dist_overhead_false = ann_dists / nn_dists



            recall_1s.append(recall)
            recall_ns.append(recall_n)
            dist_overheads_false.append(dist_overhead_false)

        column = {
            'Recall-1': f"{np.mean(recall_1s):.2f} +- {np.std(recall_1s)}",
            f'Recall-{n}': f"{np.mean(recall_ns):.2f} +- {np.std(recall_ns)}",
            'ANN-distance over NN-distance (False-only)': f"{np.mean(dist_overheads_false):.2f} +- {np.std(dist_overheads_false)}",
        }

        table[ANN.name] = pd.Series(column)
        print(column)
        table.to_csv("accuracy_table.csv")


def produce_timing_table():
    """measure the time it takes to compute NN, approximate NN and SWD on different sizes of patch sets"""
    methods = [
        swd(p, n_proj=64, use_gpu=True),
        # PytorchNN(batch_size=256, alpha=1, use_gpu=False),
        FaissFlat(use_gpu=True),
        FaissIVF(nprobe=1, n_first=1, use_gpu=True),
        # FaissIVF(nprobe=10, n_first=1, use_gpu=True),
        FaissIVFPQ(nprobe=1, n_first=1, use_gpu=True),
    ]
    table = pd.DataFrame()
    for s in [64, 128, 256, 512, 1024]:
        n = s ** 2
        d = 3 * p ** 2
        X = torch.randn((n, d))
        Y = torch.randn((n, d))

        column = {method.name: measure_funtion_call_time(method, X, Y)[0] for method in methods}

        table[s] = pd.Series(column)
        print(f"Res {s}: {column}")
        table.to_csv("timing_table.csv")

if __name__ == '__main__':
    p = 8
    n_reps = 1

    test_ann_accuracy()
    produce_timing_table()
