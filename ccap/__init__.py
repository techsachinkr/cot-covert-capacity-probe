"""CCAP — Controllable-Capacity Auditing Probe.

An open measurement harness that produces a density-estimation-free, operational
*upper bound* on the usable (controllable-and-decodable) covert-channel capacity
of an LLM's chain-of-thought, per encoding mechanism.

The intellectual move: CCAP measures *instructed* capacity. A model cannot
covertly transmit more bits through a mechanism than it can controllably transmit
when explicitly trying to. Therefore instructed controllable capacity is a
legitimate upper bound on what any emergent channel could exploit through the same
mechanism — a ceiling, which is what a safety auditor needs.

Quick start (no GPU required)::

    from ccap.runner import ExperimentRunner
    from ccap.grid import GridSpec
    from ccap.models import MockBackend

    runner = ExperimentRunner(backends={"sim-hi": MockBackend(fidelity=0.95)})
    df = runner.run(GridSpec.demo())
    # then: ccap aggregate / ccap figures
"""

from .types import (
    Mode,
    GenRequest,
    GenResult,
    SimContext,
    TrialSpec,
    TrialResult,
)

__version__ = "0.1.0"

__all__ = [
    "Mode",
    "GenRequest",
    "GenResult",
    "SimContext",
    "TrialSpec",
    "TrialResult",
    "__version__",
]
