import numpy as np
from sklearn.gaussian_process.kernels import (
    Kernel,
    NormalizedKernelMixin,
    StationaryKernelMixin,
)


class MTGPKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, kernel_f, kernel_z):
        self.kernel_f = kernel_f
        self.kernel_z = kernel_z

    #
    def set_new_params(self, sigma_f, ell_z, ell_f):
        self.kernel_f.set_params(k1__constant_value=sigma_f, k2__length_scale=ell_f)
        self.kernel_z.set_params(k2__length_scale=ell_z)

    def calc_diff(self, x1, x2):
        K_f = self.kernel_f(x1[:, 1:], x2[:, 1:])
        K_z = self.kernel_z(np.c_[x1[:, 0]], np.c_[x2[:, 0]])
        return K_f * K_z

    def diag(self, x):
        K_f = self.kernel_f.diag(x[:, 1:])
        K_z = self.kernel_z.diag(x[:, 0])
        return K_f * K_z

    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        K = self.calc_diff(x1, x2)
        return K


def optimize_scale(kernel, X, Y, beta, error_opt=True):
    error_min = kernel.kernel_z.get_params()["k2__length_scale_bounds"][0]
    error_max = kernel.kernel_z.get_params()["k2__length_scale_bounds"][1]
    ell_min = kernel.kernel_f.get_params()["k2__length_scale_bounds"][0]
    ell_max = kernel.kernel_f.get_params()["k2__length_scale_bounds"][1]
    initial_error = kernel.kernel_z.get_params()["k2__length_scale"]
    selected_sigma_f = 1
    evidence = -np.inf
    ystd = np.std(Y)
    if ystd == 0:
        ystd = 1

    Y_nor = (Y - np.mean(Y)) / ystd
    ell_range = np.exp(np.linspace(np.log(ell_min), np.log(ell_max), 100))
    error_range = np.exp(np.linspace(np.log(error_min), np.log(error_max), 10))

    if error_opt:
        for ell in ell_range:
            for error in error_range:
                kernel.set_new_params(selected_sigma_f, error, ell)
                Gram = kernel(X)
                covariance = Gram + np.identity(np.size(X, 0)) / beta
                L = np.linalg.cholesky(covariance)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_nor))
                new_evidence = -(Y_nor.T).dot(alpha) / 2 - np.sum(np.log(np.diag(L)))
                if new_evidence > evidence:
                    selected_error = error
                    selected_ell = ell
                    evidence = new_evidence
        kernel.set_new_params(selected_sigma_f, selected_error, selected_ell)
    else:
        for ell in ell_range:
            kernel.set_new_params(selected_sigma_f, initial_error, ell)
            Gram = kernel(X)
            covariance = Gram + np.identity(np.size(X, 0)) / beta
            L = np.linalg.cholesky(covariance)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_nor))
            evidence = -(Y_nor.T).dot(alpha) / 2 - np.sum(np.log(np.diag(L)))
            if new_evidence > evidence:
                selected_ell = ell
                evidence = new_evidence
            kernel.set_new_params(selected_sigma_f, initial_error, selected_ell)
