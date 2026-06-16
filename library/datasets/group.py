import logging
from typing import Any, List, Sequence

import torch
from accelerate import Accelerator

from library.anima.text_strategies import TextEncoderOutputsCachingStrategy
from library.datasets.dreambooth import DreamBoothDataset

logger = logging.getLogger(__name__)


# behave as Dataset mock
class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[DreamBoothDataset]):
        self.datasets: List[DreamBoothDataset]

        super().__init__(datasets)

        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0

        for dataset in datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    def refresh_concat_state(self):
        """Recompute ConcatDataset cumulative sizes + aggregate image_data after a
        member dataset's length changed.

        ``ConcatDataset`` caches ``cumulative_sizes`` (and we cache ``image_data`` /
        counts) at ``__init__``. A later per-dataset re-bucket — e.g.
        ``BaseDataset.restrict_to_byg_tuples`` dropping images without an edit-tuple
        sidecar — shrinks each member's ``buckets_indices`` but leaves these caches
        stale, so a global index maps to a ``sample_idx`` past the shrunken member
        and ``__getitem__`` raises ``IndexError``. Call this after any such rebuild.
        """
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0
        for dataset in self.datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    def set_text_encoder_output_caching_strategy(
        self, strategy: TextEncoderOutputsCachingStrategy
    ):
        for dataset in self.datasets:
            dataset.set_text_encoder_output_caching_strategy(strategy)

    def enable_XTI(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.enable_XTI(*args, **kwargs)

    def cache_latents(
        self,
        vae,
        vae_batch_size=1,
        cache_to_disk=False,
        is_main_process=True,
        file_suffix=".npz",
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_latents(
                vae, vae_batch_size, cache_to_disk, is_main_process, file_suffix
            )

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_latents(model, accelerator)
        accelerator.wait_for_everyone()

    def cache_text_encoder_outputs(
        self,
        tokenizers,
        text_encoders,
        device,
        weight_dtype,
        cache_to_disk=False,
        is_main_process=True,
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs(
                tokenizers,
                text_encoders,
                device,
                weight_dtype,
                cache_to_disk,
                is_main_process,
            )

    def cache_text_encoder_outputs_sd3(
        self,
        tokenizer,
        text_encoders,
        device,
        output_dtype,
        te_dtypes,
        cache_to_disk=False,
        is_main_process=True,
        batch_size=None,
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs_sd3(
                tokenizer,
                text_encoders,
                device,
                output_dtype,
                te_dtypes,
                cache_to_disk,
                is_main_process,
                batch_size,
            )

    def new_cache_text_encoder_outputs(
        self, models: List[Any], accelerator: Accelerator
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_text_encoder_outputs(models, accelerator)
        accelerator.wait_for_everyone()

    def set_caching_mode(self, caching_mode):
        for dataset in self.datasets:
            dataset.set_caching_mode(caching_mode)

    def is_latent_cacheable(self) -> bool:
        return all([dataset.is_latent_cacheable() for dataset in self.datasets])

    def is_latents_cache_complete(self) -> bool:
        return all(dataset.is_latents_cache_complete() for dataset in self.datasets)

    def is_text_encoder_outputs_cache_complete(self) -> bool:
        return all(
            dataset.is_text_encoder_outputs_cache_complete()
            for dataset in self.datasets
        )

    def count_repa_pe_sidecars(self) -> tuple[int, int]:
        present = 0
        total = 0
        for dataset in self.datasets:
            p, t = dataset.count_repa_pe_sidecars()
            present += p
            total += t
        return (present, total)

    def is_text_encoder_output_cacheable(
        self, cache_supports_dropout: bool = False
    ) -> bool:
        return all(
            [
                dataset.is_text_encoder_output_cacheable(cache_supports_dropout)
                for dataset in self.datasets
            ]
        )

    def set_current_strategies(self):
        for dataset in self.datasets:
            dataset.set_current_strategies()

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        for dataset in self.datasets:
            dataset.disable_token_padding()
