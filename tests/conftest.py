"""scipy 1.17 removed `simps`; SF 0.23 still imports it (only used in Wigner
integration paths we don't touch). Shim before SF gets imported by any test."""

import scipy.integrate as _si

if not hasattr(_si, "simps"):
    _si.simps = _si.simpson
