from .l1 import L1Slow
from .l21 import L21Slow
from .squaredl12 import SquaredL12Slow
from .squaredl21 import SquaredL21Slow
from .omegati import OmegaTISlow
from .omegacs import OmegaCSSlow

REGULARIZATION = {
    'squaredl12': SquaredL12Slow,
    'squaredl21': SquaredL21Slow,
    'l1': L1Slow,
    'l21': L21Slow,
    'omegati': OmegaTISlow,
    'omegacs': OmegaCSSlow
}
