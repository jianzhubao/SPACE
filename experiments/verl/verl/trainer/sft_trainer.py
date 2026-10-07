# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from functools import partial

from tensordict.tensorclass import NonTensorData

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.distributed import destroy_global_process_group
from verl.utils.logger import log_with_rank
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric import Metric
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.tracking import Tracking
from verl.workers.engine_workers import TrainingWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def _mean_metric_value(value) -> float:
    if isinstance(value, Metric):
        return float(value.aggregate())
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item())
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, Metric) for item in value):
            return float(Metric.aggregate_dp(value))
        values = [_mean_metric_value(item) for item in value]
        return sum(values) / len(values) if values else 0.0
    return float(value)


def _resolve_loss_function(loss_type: str):
    from verl.workers.utils.losses import dft_loss, sft_loss

    if loss_type == "asft":
        from space.trainer.asft import asft_loss

        return asft_loss
    if loss_type == "psft":
        from space.trainer.psft import psft_loss

        return psft_loss
    loss_functions = {"sft": sft_loss, "dft": dft_loss}
    if loss_type not in loss_functions:
        raise ValueError(f"Unsupported trainer.loss_type={loss_type!r}; expected one of ['asft', 'dft', 'psft', 'sft']")
    return loss_functions[loss_type]


class SFTTrainer:
    def __init__(
        self,
        config,
    ):
        self.config = config

        log_gpu_memory_usage(f"rank {torch.distributed.get_rank()}: Before SFTTrainer init", logger=logger)

        self.rank = torch.distributed.get_rank()

        self._build_config()
        self._build_dataset()

        self._build_engine()

        self._build_dataloader()

        self._init_engine()

        self.online_projection = self._build_online_projection()
        self._validate_additional_loss_selection()
        self.sub_loss = self._build_sub_loss()
        self.rpsft_loss = self._build_rpsft_loss()
        self.additional_loss = self.rpsft_loss or self.sub_loss
        self.engine.set_additional_loss_fn(
            self._compute_additional_loss if self.additional_loss is not None else None
        )

        self._build_ckpt_handler()

        # Initialize resume-related variables
        self.resume_global_step = self.ckpt_handler.load_checkpoint()

        self.device_name = self.config.trainer.device

        if self.rank == 0:
            print(self.config)

        log_gpu_memory_usage(f"rank {self.rank}: After SFTTrainer init", logger=logger)

    def _build_ckpt_handler(self):
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)
        lora_train_meta = self._get_lora_train_meta()

        self.ckpt_handler = CheckpointHandler(
            engine=self.engine,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
            lora_train_meta=lora_train_meta,
        )

    def _build_online_projection(self):
        raw_config = self.config.get("online_proj", None)
        if raw_config is None or not bool(raw_config.get("enabled", False)):
            return None

        from space.trainer.online_proj import build_online_projection

        return build_online_projection(
            raw_config,
            default_base_model=self.config.model.path,
            model=self.engine.module,
            lora_rank=int(getattr(self.model_config, "lora_rank", 0) or 0),
            output_dir=self.config.trainer.default_local_dir,
            rank=self.rank,
        )

    def _maybe_apply_online_projection(self, step: int, *, is_last_step: bool) -> None:
        if self.online_projection is not None:
            self.online_projection.apply(model=self.engine.module, step=step, is_last_step=is_last_step)

    def _build_sub_loss(self):
        raw_config = self.config.get("sub_loss", None)
        if raw_config is None or not bool(raw_config.get("enabled", False)):
            return None

        from space.trainer.sub_loss import build_subspace_loss

        return build_subspace_loss(
            raw_config,
            default_base_model=self.config.model.path,
            model=self.engine.module,
            lora_rank=int(getattr(self.model_config, "lora_rank", 0) or 0),
            output_dir=self.config.trainer.default_local_dir,
            rank=self.rank,
        )

    def _validate_additional_loss_selection(self) -> None:
        sub_loss_config = self.config.get("sub_loss", None)
        rpsft_config = self.config.get("rpsft", None)
        sub_loss_enabled = sub_loss_config is not None and bool(sub_loss_config.get("enabled", False))
        rpsft_enabled = rpsft_config is not None and bool(rpsft_config.get("enabled", False))
        if sub_loss_enabled and rpsft_enabled:
            raise ValueError("sub_loss and rpsft cannot be enabled at the same time.")

    def _build_rpsft_loss(self):
        raw_config = self.config.get("rpsft", None)
        if raw_config is None or not bool(raw_config.get("enabled", False)):
            return None

        from space.trainer.rpsft_loss import build_rpsft_loss

        return build_rpsft_loss(
            raw_config,
            default_base_model=self.config.model.path,
            model=self.engine.module,
            lora_rank=int(getattr(self.model_config, "lora_rank", 0) or 0),
            output_dir=self.config.trainer.default_local_dir,
            rank=self.rank,
        )

    def _compute_additional_loss(self, data):
        step = int(tu.get(data, key="global_step"))
        return self.additional_loss.compute(model=self.engine.module, step=step)

    def _get_lora_train_meta(self):
        lora_adapter_path = self.config.model.get("lora_adapter_path", None)
        lora_rank = int(getattr(self.config.model, "lora_rank", 0) or 0)

        if lora_adapter_path is None and lora_rank <= 0:
            return None

        raw_lora_alpha = self.config.model.get("lora_alpha", None)
        if raw_lora_alpha is None:
            log_with_rank(
                "LoRA is enabled but `model.lora_alpha` is not set; fallback to 0 in checkpoint metadata.",
                logger=logger,
                rank=self.rank,
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            lora_alpha = 0
        else:
            lora_alpha = int(raw_lora_alpha)
            if lora_alpha == 0:
                log_with_rank(
                    "LoRA is enabled but `model.lora_alpha` is 0; this may lead to ineffective LoRA scaling.",
                    logger=logger,
                    rank=self.rank,
                    level=logging.WARNING,
                    log_only_rank_0=True,
                )

        task_type = self.config.model.get("task_type", None)
        if task_type is None:
            task_type = "CAUSAL_LM"

        return {
            "r": lora_rank,
            "lora_alpha": int(lora_alpha or 0),
            "task_type": str(task_type),
        }

    def _build_config(self):
        from verl.utils.config import omega_conf_to_dataclass

        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
        self.profiler_config = omega_conf_to_dataclass(self.config.profiler)
        self.loss_function = _resolve_loss_function(self.config.trainer.loss_type)
        if self.config.trainer.loss_type == "psft":
            mini_batch_size = self.config.trainer.psft_mini_batch_size
            if mini_batch_size <= 0 or self.config.data.train_batch_size % mini_batch_size:
                raise ValueError("trainer.psft_mini_batch_size must be positive and divide data.train_batch_size.")
        if self.config.trainer.loss_type == "asft" and self.config.trainer.asft_kl_coef != 0:
            if (
                self.engine_config.strategy != "fsdp2"
                or self.config.trainer.device != "cuda"
                or self.engine_config.ulysses_sequence_parallel_size != 1
                or self.model_config.lora_rank > 0
                or self.model_config.lora_adapter_path is not None
                or self.config.data.pad_mode != "no_padding"
                or not self.model_config.use_remove_padding
                or self.model_config.use_fused_kernels
            ):
                raise ValueError("ASFT requires CUDA FSDP2, SP=1, full parameters, no padding, and unfused logits.")

        # check profile interval
        self.profiler_interval = self.config.trainer.profile_interval
        self._validate_profiler_interval()

    def _validate_profiler_interval(self):
        assert len(self.profiler_interval) == 2
        self.start_profile_step = self.profiler_interval[0]
        self.end_profile_step = self.profiler_interval[1]
        assert self.end_profile_step >= self.start_profile_step
        if self.start_profile_step < 0:
            assert self.end_profile_step < 0

    def _build_engine(self):
        from verl.workers.engine_workers import TrainingWorkerConfig

        loss_config = self.config.trainer if self.config.trainer.loss_type in ("asft", "psft") else None
        self.loss_fn = partial(self.loss_function, config=loss_config)

        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        self.training_client = TrainingWorker(config=config)
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        # Note that in SPMD world, this abstraction has to break
        self.engine = self.training_client.engine

    def _init_engine(self):
        # patch optimizer config
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.steps_per_epoch = len(self.train_dataloader)

        # manage save and test frequency
        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

        self.training_client.reset()
        if self.config.trainer.loss_type == "asft" and self.config.trainer.asft_kl_coef != 0:
            from space.trainer.asft import build_asft_reference
            from verl.utils.device import get_device_id

            self.engine.asft_reference = build_asft_reference(self.model_config, get_device_id())

    def _build_dataset(self):
        config = self.config
        tokenizer = self.model_config.tokenizer
        processor = self.model_config.processor
        train_dataset = create_sft_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        if config.data.val_files:
            val_dataset = create_sft_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
        else:
            val_dataset = None

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def _build_dataloader(self):
        # build dataset
        config = self.config
        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=dp_size,
            rank=dp_rank,
            drop_last=True,
            seed=config.trainer.seed,
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            val_drop_last = bool(config.data.get("val_drop_last", True))
            self.val_sampler = DistributedSampler(
                self.val_dataset,
                shuffle=False,
                num_replicas=dp_size,
                rank=dp_rank,
                drop_last=val_drop_last,
                seed=config.trainer.seed,
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.data.num_workers,
                pin_memory=False,
                drop_last=val_drop_last,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

    def _get_batch_seqlens(self, data):
        # mean over dp group
        is_nested = data["input_ids"].is_nested
        if is_nested:
            batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()
        else:
            batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)
        batch_seqlens = batch_seqlens.to(self.device_name)  # (global_bsz // dp)
        if self.engine.get_data_parallel_size() > 1:
            output_tensor = torch.empty(
                (batch_seqlens.shape[0] * self.engine.get_data_parallel_size(),),
                dtype=batch_seqlens.dtype,
                device=self.device_name,
            )  # (global_bsz,)

        dp_group = self.engine.get_data_parallel_group()
        dp_size = self.engine.get_data_parallel_size()

        if dp_size == 1 or dp_group is None:
            return batch_seqlens.tolist()

        output_tensor = torch.empty(
            (batch_seqlens.shape[0] * dp_size,),
            dtype=batch_seqlens.dtype,
            device=self.device_name,
        )  # (global_bsz,)

        torch.distributed.all_gather_into_tensor(
            output_tensor=output_tensor,
            input_tensor=batch_seqlens,
            group=dp_group,
        )

        batch_seqlens = output_tensor.tolist()
        return batch_seqlens

    def fit(self):
        is_logging = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0

        # TODO: add a unified tracking
        if is_logging:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        train_time = 0
        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            aggressive_empty_cache(force_sync=True)
            log_gpu_memory_usage(f"rank {self.rank}: At start of epoch {epoch}", logger=logger)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                # construct tensordict
                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                batch_seqlens = self._get_batch_seqlens(data=data)
                # this is necessary. Otherwise, it is interpreted as NonTensorStack
                batch_seqlens_ntd = NonTensorData(batch_seqlens)

                tu.assign_non_tensor(
                    data,
                    update_lr_scheduler=True,
                    global_token_num=batch_seqlens_ntd,
                    global_step=global_step,
                )

                # start profile in SPMD mode
                if global_step == self.start_profile_step:
                    self.training_client.start_profile()
                if self.config.trainer.loss_type == "psft":
                    from space.trainer.psft import train_psft_batch

                    output = train_psft_batch(self.training_client, data, self.config.trainer.psft_mini_batch_size)
                else:
                    output = self.training_client.train_batch(data=data)

                if global_step == self.end_profile_step:
                    self.training_client.stop_profile()

                if self.engine.is_mp_src_rank_with_outputs():
                    metrics = tu.get(output, "metrics")
                    if self.config.trainer.loss_type == "psft":
                        metrics = {
                            key: max(value) if key.startswith("perf/max_memory_") else _mean_metric_value(value)
                            for key, value in metrics.items()
                        }

                    # TODO: we can actual accumulate metrics for N steps and perform aggregate metrics
                    for k in ["loss", "grad_norm", "lr", "mfu"]:
                        if k in metrics.keys():
                            value = metrics.pop(k)
                            metrics[f"train/{k}"] = value

                    for key in ("original_loss", "dft_loss", "kl_loss", "psft_clip_fraction", "psft_approx_kl"):
                        if key in metrics:
                            metrics[f"train/{key}"] = _mean_metric_value(metrics.pop(key))

                    for key in (
                        "sub_loss",
                        "sub_loss_scaled",
                        "sub_loss_active_targets",
                        "sub_loss_mean_rank",
                        "sub_loss_applied",
                        "rpsft_loss",
                        "rpsft_loss_scaled",
                        "rpsft_active_targets",
                        "rpsft_mean_rank",
                        "rpsft_full_rank_targets",
                        "rpsft_prepare_seconds",
                        "rpsft_step_seconds",
                        "rpsft_applied",
                    ):
                        if key in metrics:
                            metrics[f"train/{key}"] = _mean_metric_value(metrics.pop(key))

                    metrics["train/global_tokens"] = torch.sum(
                        torch.tensor(batch_seqlens, device=self.device_name)
                    ).item()
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9

                    if self.engine.get_data_parallel_rank() == 0:
                        tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0
                should_save_checkpoint = is_last_step or (self.save_freq > 0 and is_save_step)

                if self.online_projection is not None and should_save_checkpoint:
                    aggressive_empty_cache(force_sync=True)
                    self.online_projection.save_before_projection_checkpoint(
                        engine=self.engine,
                        step=global_step,
                        is_last_step=is_last_step,
                    )

                self._maybe_apply_online_projection(global_step, is_last_step=is_last_step)

                # early exit or validation step
                if is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    val_aux_losses = {key: [] for key in ("original_loss", "dft_loss", "kl_loss")}
                    for val_data in self.val_dataloader:
                        val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
                        output = self.training_client.infer_batch(val_data)

                        if self.engine.is_mp_src_rank_with_outputs():
                            metrics = tu.get(output, "metrics")
                            val_losses.append(metrics["loss"])
                            for key, values in val_aux_losses.items():
                                if key in metrics:
                                    values.append(_mean_metric_value(metrics[key]))

                    if self.engine.is_mp_src_rank_with_outputs():
                        val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))
                        # average over data parallel group
                        dp_group = self.engine.get_data_parallel_group()
                        if dp_group is not None:
                            torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)

                    if is_logging:
                        metric = {"val/loss": val_loss.detach().item()}
                        for key, values in val_aux_losses.items():
                            if values:
                                metric[f"val/{key}"] = sum(values) / len(values)
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if should_save_checkpoint:
                    aggressive_empty_cache(force_sync=True)
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    if is_logging:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    trainer = SFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_object

        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        # Default to multi-turn dataset
        dataset_cls = MultiTurnSFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(
        parquet_files=data_paths, tokenizer=tokenizer, config=data_config, processor=processor, max_samples=max_samples
    )
    return dataset


if __name__ == "__main__":
    main()
