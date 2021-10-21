from .batched import Batched
from .continuous import Continuous

KNOWN_ALGORITHMS = {'batched': Batched,
                    'continuous': Continuous
                    }
