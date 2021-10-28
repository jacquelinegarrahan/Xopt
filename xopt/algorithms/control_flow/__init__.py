from .batched import run_batched
from .continuous import run_continuous

KNOWN_CONTROL_FLOW = {"batched": run_batched, "continuous": run_continuous}
