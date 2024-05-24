import math
from functools import lru_cache
from typing import Iterator, Tuple

import line_profiler
import numpy as np
import scipy
import torch
from torch import nn
from torch.utils.data import IterableDataset, get_worker_info


class VonKarmanXY(IterableDataset):
    def __init__(
        self,
        num_samples: int,
        resolution: Tuple[int, int],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        L0_vals: np.ndarray,
        L0_probs: np.ndarray | None = None,
        vk_cache: bool = True,
        base_seed: int = 0,
        tfms: torch.nn.Module | None = None,
    ):
        self.num_samples = num_samples
        self.res_x, self.res_y = resolution
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.L0_vals = L0_vals
        self.L0_probs = L0_probs
        self.vk_cache = vk_cache
        self.base_seed = base_seed
        self.tfms = tfms

        if self.L0_probs is None:
            self.L0_probs = np.array([1 / len(self.L0_vals)] * len(self.L0_vals))
        else:
            assert len(self.L0_vals) == len(
                self.L0_probs
            ), "The number of L0 values must match the number of probabilities."
            assert math.isclose(
                sum(self.L0_probs), 1
            ), "The probabilities must sum to 1."

        self.__rho = self.__create_rho_matrix()

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, float]]:
        worker_info = get_worker_info()
        if worker_info is not None:
            # for multiprocessing: set the numpy seed based on the worker id
            worker_id: int = worker_info.id
            np.random.seed(self.base_seed + worker_id)

        for _ in range(self.num_samples):
            L0 = self.__sample_L0()

            # get the cholesky factor for the VonKarman covariance matrix
            # only use caching if the flag is set. This logic was added
            # for performance testing
            if self.vk_cache:
                cholesky = self.create_vonkarman_cholesky_factor(L0)
            else:
                cholesky = self.create_vonkarman_cholesky_factor.__wrapped__(self, L0)
            cholesky = self.create_vonkarman_cholesky_factor(L0)

            # draw a sample
            b = np.random.randn(self.res_x * self.res_y)
            n = cholesky @ b

            # reshape to have dimensions of height x width
            # add dummy dimensions for batch and channel
            # batch is required by transforms and channel is required for automatic abtching
            n = n.reshape(self.res_x, self.res_y)[None, None]

            n_pt = torch.tensor(n).type(torch.float32)
            if self.tfms is not None:
                n_pt = self.tfms(n_pt)
            # remove the dummy batch dimension
            n_pt = n_pt.squeeze(0)

            yield n_pt, L0

    def __len__(self):
        return self.num_samples

    @lru_cache(maxsize=None)
    def create_vonkarman_cholesky_factor(self, L0: float) -> np.ndarray:
        """
        Create the von Karman precision matrix.

        Returns:
            torch.Tensor: The von Karman precision matrix.
        """

        # compute the vonkarman covariance matrix up to a multiplicative constant
        # avoid repeating computation for the same value of __rho
        vk_cov = scipy.special.kv(
            5 / 6, 2 * math.pi * self.__rho / L0
        ) * self.__rho ** (5 / 6)

        # compute the cholesky factor of the vonkarman covariance matrix
        return np.linalg.cholesky(vk_cov)

    def __sample_L0(self) -> float:
        return np.random.choice(self.L0_vals, p=self.L0_probs)

    def __create_rho_matrix(self) -> np.ndarray:
        x_vals = torch.linspace(self.x_min, self.x_max, self.res_x)
        y_vals = torch.linspace(self.y_min, self.y_max, self.res_y)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
        X, Y = X.flatten(), Y.flatten()

        XY = torch.stack((X, Y), dim=-1)

        # compute the distance between all pairs of points
        rho = torch.sqrt(torch.mean((XY[:, None] - XY[None]) ** 2, dim=-1))

        # purturb zero values away from zero for correct rho -> 0 limit behavior in the von Karman matrix.
        rho[rho == 0] = 1e-10

        rho = rho.cpu().numpy()

        return rho
    
    def __repr__(self) -> str:
        return f"VonKarmanXY(num_samples={self.num_samples}, resolution={self.res_x}x{self.res_y}, x_range=({self.x_min}, {self.x_max}), y_range=({self.y_min}, {self.y_max}), L0_vals={self.L0_vals}, L0_probs={self.L0_probs}, vk_cache={self.vk_cache}, base_seed={self.base_seed}, tfms={repr(self.tfms)})"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tfms: nn.Module = nn.Upsample(size=(64, 64), mode="nearest")

    dataset = VonKarmanXY(
        num_samples=10,
        resolution=(24, 24),
        x_range=(-1, 1),
        y_range=(-1, 1),
        L0_vals=np.logspace(-1, 1, 10),
        L0_probs=None,
        vk_cache=False,
        tfms=tfms,
    )

    profile = line_profiler.LineProfiler()
    profile.add_function(dataset.create_vonkarman_cholesky_factor.__wrapped__)
    profile.enable()

    # Run the code
    for n, L0 in dataset:
        # plot the density distribution
        plt.imshow(n.squeeze())
        plt.title(str(L0))

        # save the plot
        plt.savefig(f"plots/vk_{L0}.png")

    profile.disable()
    profile.print_stats()
