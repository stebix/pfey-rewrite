import importlib
import numpy as np
import torch



def get_nonlinearity(nonlinearity: str, **kwargs) -> torch.nn.Module:
    module = importlib.import_module(name='torch.nn')
    nonlinearty_class = getattr(module, nonlinearity)
    return nonlinearty_class(**kwargs)


def recursive_to_tensor(candidate):
    if isinstance(candidate, (list, tuple)):
        return [recursive_to_tensor(elem) for elem in candidate]
    elif isinstance(candidate, np.ndarray):
        return torch.from_numpy(candidate)
    elif isinstance(candidate, torch.Tensor):
        return candidate
    else:
        message = f'could not cast type {type(candidate)} to tensor'
        raise RuntimeError(message)




if __name__ == '__main__':
    _ = get_nonlinearity('ReLU')