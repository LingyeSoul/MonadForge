"""Training-free sampler-boundary corrections that plug into the denoise loop.

Each module is an optional technique applied at (or around) the sampler step,
not part of the core engine: SMC-CFG (``smc_cfg``), DCW SNR-t bias correction
(``dcw_calibrator``), and mod-guidance AdaLN (``mod_guidance``). Imported
on demand by ``library.inference.generation``.
"""
