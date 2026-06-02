import torch

from bench.freetext.stage2 import SGMIInjector


def test_sgmi_injector_preserves_5d_sampler_shape():
    z_next = torch.zeros(1, 16, 1, 4, 5, dtype=torch.bfloat16)
    z_ref = torch.ones(1, 16, 4, 5)
    mask = torch.ones(1, 1, 4, 5)

    injector = SGMIInjector(
        z_ref=z_ref,
        mask=mask,
        sigma_start=0.8,
        sigma_end=0.6,
        anneal="flat",
        use_log_gabor=False,
        seed=0,
    )

    out = injector.apply(z_next, sigma_next=0.7)

    assert out.shape == z_next.shape
    assert out.dtype == z_next.dtype
    assert injector.log == [{"sigma": 0.7, "lam": 1.0}]


def test_sgmi_injector_preserves_4d_latent_shape():
    z_next = torch.zeros(1, 16, 4, 5, dtype=torch.bfloat16)
    z_ref = torch.ones(1, 16, 4, 5)
    mask = torch.ones(1, 1, 4, 5)

    injector = SGMIInjector(
        z_ref=z_ref,
        mask=mask,
        sigma_start=0.8,
        sigma_end=0.6,
        anneal="flat",
        use_log_gabor=False,
        seed=0,
    )

    out = injector.apply(z_next, sigma_next=0.7)

    assert out.shape == z_next.shape
    assert out.dtype == z_next.dtype
