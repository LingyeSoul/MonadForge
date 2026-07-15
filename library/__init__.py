from library.runtime.compat import configure_bitsandbytes_cuda_override


# bitsandbytes releases can lag a new PyTorch CUDA minor. Select its newest
# bundled binary in the same CUDA major before any indirect bitsandbytes import.
configure_bitsandbytes_cuda_override()

del configure_bitsandbytes_cuda_override
