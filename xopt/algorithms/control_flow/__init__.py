from .continuous import run_continuous
from .batched import run_batched

KNOWN_CONTROL_FLOW = {"batched": run_batched, "continuous": run_continuous}
