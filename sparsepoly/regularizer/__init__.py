from .l1 import L1
from .l21 import L21
from .omegacs import OmegaCS
from .omegati import OmegaTI
from .squaredl12 import SquaredL12
from .squaredl21 import SquaredL21

REGULARIZATION = {
    'squaredl12': SquaredL12,
    'squaredl21': SquaredL21,
    'l1': L1,
    'l21': L21,
    'omegati': OmegaTI,
    'omegacs': OmegaCS
}
