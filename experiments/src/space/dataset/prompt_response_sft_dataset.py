"""Response-only SFT dataset compatible with Qwen3 thinking responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils import hf_tokenizer
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.py_functional import convert_nested_value_to_list_recursive


class PromptResponseSFTDataset(Dataset):
    """Build prompt-template plus raw-response sequences for text-only SFT."""

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: ProcessorMixin | None = None,
        max_samples: int = -1,
    ) -> None:
        config = config or {}
        self.pad_mode = config.get("pad_mode", DatasetPadMode.NO_PADDING)
        if self.pad_mode != DatasetPadMode.NO_PADDING:
            raise ValueError(
                "PromptResponseSFTDataset supports only pad_mode=no_padding, "
                f"got {self.pad_mode!r}"
            )
        self.max_length = int(config.get("max_length", 1024))
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        self.truncation = config.get("truncation", "error")
        if self.truncation not in {"error", "left", "right"}:
            raise ValueError(f"Unknown truncation method {self.truncation!r}")

        self.messages_key = config.get("messages_key", "messages")
        self.prompt_key = config.get("prompt_key", "message")
        self.response_key = config.get("response_key", "response")
        self.apply_chat_template_kwargs = dict(
            config.get("apply_chat_template_kwargs", {})
        )
        self.strip_response_thinking_prefix = bool(
            config.get("strip_response_thinking_prefix", False)
        )
        self.max_samples = max_samples

        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]
        self.parquet_files = [
            copy_local_path_from_hdfs(path, verbose=True) for path in parquet_files
        ]

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id must be defined")

        self._read_files()

    def _read_files(self) -> None:
        dataframes = [
            pd.read_parquet(path, dtype_backend="pyarrow")
            for path in self.parquet_files
        ]
        self.dataframe = pd.concat(dataframes, ignore_index=True)
        if self.max_samples > 0:
            self.dataframe = self.dataframe.iloc[: self.max_samples]

        has_separate_fields = {
            self.prompt_key,
            self.response_key,
        }.issubset(self.dataframe.columns)
        has_messages = self.messages_key in self.dataframe.columns
        if not has_separate_fields and not has_messages:
            raise KeyError(
                "dataset must contain either separate "
                f"{self.prompt_key!r}/{self.response_key!r} columns or a "
                f"{self.messages_key!r} column"
            )

        self.prompts: list[list[dict[str, Any]]] = []
        self.responses: list[str] = []
        for row_index, row in self.dataframe.iterrows():
            if has_separate_fields:
                prompt = row[self.prompt_key]
                response = row[self.response_key]
            else:
                messages = self._as_messages(
                    row[self.messages_key], row_index, self.messages_key
                )
                if len(messages) < 2 or messages[-1].get("role") != "assistant":
                    raise ValueError(
                        f"Row {row_index}: {self.messages_key!r} must end with "
                        "an assistant response and contain at least one prompt turn"
                    )
                prompt = messages[:-1]
                response = messages[-1].get("content")

            prompt = self._as_messages(prompt, row_index, self.prompt_key)
            if not prompt:
                raise ValueError(f"Row {row_index}: prompt must not be empty")
            if not isinstance(response, str) or not response:
                raise ValueError(
                    f"Row {row_index}: response must be a non-empty string"
                )
            self.prompts.append(prompt)
            self.responses.append(response)

    @staticmethod
    def _as_messages(value: Any, row_index: int, key: str) -> list[dict[str, Any]]:
        value = convert_nested_value_to_list_recursive(value)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"Row {row_index}: {key!r} must be a message sequence")

        messages = []
        for message_index, message in enumerate(value):
            if not isinstance(message, Mapping):
                raise TypeError(
                    f"Row {row_index}: {key!r}[{message_index}] must be a mapping"
                )
            message = dict(message)
            if not isinstance(message.get("role"), str) or "content" not in message:
                raise ValueError(
                    f"Row {row_index}: {key!r}[{message_index}] must contain "
                    "role and content"
                )
            messages.append(message)
        return messages

    @staticmethod
    def _input_ids(encoded: Any, field: str) -> torch.Tensor:
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if not isinstance(encoded, torch.Tensor):
            encoded = torch.as_tensor(encoded)
        if encoded.ndim == 2:
            if encoded.shape[0] != 1:
                raise ValueError(
                    f"{field} tokenization must have batch size 1, got {encoded.shape}"
                )
            encoded = encoded[0]
        if encoded.ndim != 1:
            raise ValueError(
                f"{field} tokenization must be one-dimensional, got {encoded.shape}"
            )
        return encoded.to(dtype=torch.long)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        prompt_ids = self._input_ids(
            self.tokenizer.apply_chat_template(
                self.prompts[item],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                **self.apply_chat_template_kwargs,
            ),
            "prompt",
        )
        response = self.responses[item]
        if self.strip_response_thinking_prefix:
            for prefix in ("<think>\r\n", "<think>\n", "<think>"):
                if response.startswith(prefix):
                    response = response[len(prefix) :]
                    break

        response_ids = self._input_ids(
            self.tokenizer(
                response,
                add_special_tokens=False,
                return_tensors="pt",
            ),
            "response",
        )
        eos = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)

        input_ids = torch.cat((prompt_ids, response_ids, eos))

        loss_mask = torch.cat(
            (
                torch.zeros_like(prompt_ids),
                torch.ones(response_ids.numel() + 1, dtype=torch.long),
            )
        )

        if input_ids.numel() > self.max_length:
            if self.truncation == "error":
                raise ValueError(
                    f"sequence_length={input_ids.numel()} is larger than "
                    f"max_length={self.max_length}"
                )
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            else:
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]

        if not torch.any(loss_mask):
            raise ValueError(
                f"Row {item}: truncation removed every supervised response token"
            )
        position_ids = torch.arange(input_ids.numel(), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
