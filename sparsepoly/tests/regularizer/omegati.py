import numpy as np

class OmegaTISlow(object):
    def __init__(self):
        pass

    def _eval(self, Ps, degree):
        n_features, n_components = Ps[0].shape
        cache = np.zeros((degree+1, n_components))
        result = []
        for P in Ps:
            cache[:, :] = 0
            cache[0, :] = 1.0
            for j in range(n_features):
                abs_p_j  = abs(P[j])
                for deg in range(degree):
                    cache[degree-deg] += cache[degree-deg-1] * abs_p_j
            result.append(np.sum(cache[degree]))
        return np.array(result)
    
    def _eval_all(self, Ps):
        return np.sum(np.sum(np.prod(np.abs(Ps)+1.0, axis=1), 0))

    def eval(self, P, degree):
        shape = P.shape[:-2]
        if degree > 0:
            result = self._eval(P.reshape(-1, P.shape[-2], P.shape[-1]), degree)
            if P.ndim == 2:
                return result[0]
            else:
                return result.reshape(shape)
        elif degree == -1:
            return self._eval_all(P.reshape(-1, P.shape[-2], P.shape[-1]))
        else:
            raise ValueError("degree must be a positive int or -1 (all).")

    def prox_cd(self, p_sj, ps, strength, degree, j):
        sign = 1 if p_sj > 0 else -1

        if degree > 0: # factorization machine
            cache = np.zeros(degree)
            cache[0] = 1.0
            for j1 in range(len(ps)):
                if j1 != j:
                    for t in range(degree-1):
                        cache[degree-t-1] += cache[degree-t-2] * abs(ps[j1])
            strength *= cache[-1]
        else: # all-subsets
            cache = 1.0
            for j1 in range(len(ps)):
                if j1 != j: 
                    cache *= (1.0 + abs(ps[j1]))
            strength *= cache

        return sign * np.maximum(abs(p_sj) - strength, 0)
