import numpy as np

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ''
        for key in self.__dict__.keys():
            s = s + key + ':' + 1*'\t' + str( self.__dict__[key] ) + '\n'

        return s

def one_hot(a, M):
    onehot = np.zeros(M)
    onehot[a] = 1
    return onehot


def db_to_lineal(x):
    return 10 ** (x / 10)


def _get_norm_factor(x, P_M):
    return np.sum(P_M * np.abs(np.power(x, 2)))


def get_norm_qam(M, P, P_M):
    r = np.arange(np.array(np.sqrt(M)))
    r = 2 * (r - np.mean(r))
    r = np.meshgrid(r, r)
    const = np.expand_dims(np.reshape(r[0] + 1j * r[1], [-1]), axis=0)
    norm_factor = np.sqrt(P / _get_norm_factor(const, P_M))
    norm_constellation = norm_factor * const
    avg_power = calculate_avg_power(norm_constellation, P_M)
    return norm_constellation

def normalize(const, P, P_M):
    norm_factor = np.sqrt(P / _get_norm_factor(const, P_M))
    norm_constellation = norm_factor * const
    avg_power = calculate_avg_power(norm_constellation, P_M)
    return norm_constellation


def calculate_avg_power(x, P_M):
    return np.sum(np.power(np.abs(x),2)*P_M)


def sampler(P_M, B):
    samples = np.zeros((np.sum(np.rint(B*P_M).astype(int)),0))
    for idx, p in enumerate(P_M):
        occurrences = np.rint(B*p).astype(int)
        samples = np.append(samples, np.ones(occurrences)*idx)
    np.random.shuffle(samples)
    return samples.astype(int)

def softmax(x):
    return np.exp(x)/sum(np.exp(x))