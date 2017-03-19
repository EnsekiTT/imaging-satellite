import numpy as np

class SOM():
    def __init__(self, vectors, N, seed=None):
        vectors = np.array(vectors)
        self.N = N
        self.sigma0 = self.N/2.0
        self.lamb = 0.5
        if not seed is None:
            np.random.seed(seed)

        x, y = np.meshgrid(range(self.N), range(self.N))
        self.c = np.hstack((x.flatten()[:, np.newaxis],
                            y.flatten()[:, np.newaxis]))
        self.W = np.random.rand(self.N*self.N,
                                    vectors.shape[1])

    def learn(self, vectors):
        self.lamb = vectors.shape[0]/4.0
        for i, vector in enumerate(vectors):
            bmu = self._BMU(vector)
            d = np.linalg.norm(self.c - bmu, axis=1)
            L = self._L(i)
            S = self._Theta(d, i)
            self.W += L * S[:, np.newaxis] * (vector - self.W)
        return self.W

    def _BMU(self, vector):
        dists = np.linalg.norm(self.W - vector, axis=1)
        bmu = np.argmin(dists)
        return np.unravel_index(bmu,(self.N, self.N))

    def _L(self, t):
        L0  = 0.1
        return L0*np.exp(-t/self.lamb)

    def _Theta(self, d, t):
        sigma = self.sigma0*np.exp(-t/self.lamb)
        return np.exp(-d**2/(2*sigma**2))
