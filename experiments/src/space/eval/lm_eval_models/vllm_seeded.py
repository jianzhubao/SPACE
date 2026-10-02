"""vLLM adapter with deterministic, independent per-request sampling streams."""

from __future__ import annotations

from copy import deepcopy

from lm_eval.api.registry import register_model
from lm_eval.models.vllm_causallms import VLLM
from vllm import SamplingParams


class _GeneratedText(str):
    """Generated text carrying its exact vLLM output-token count."""

    def __new__(cls, value: str, generated_token_count: int):
        instance = super().__new__(cls, value)
        instance.generated_token_count = generated_token_count
        return instance

    def split(self, sep=None, maxsplit=-1):
        return [
            type(self)(part, self.generated_token_count)
            for part in super().split(sep, maxsplit)
        ]

    def lstrip(self, chars=None):
        return type(self)(super().lstrip(chars), self.generated_token_count)


@register_model("vllm_seeded")
class SeededVLLM(VLLM):
    """Assign each generation request a reproducible, distinct sampling seed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampling_request_index = 0

    def _model_generate(
        self,
        requests: list[list[int]],
        generate: bool = False,
        sampling_params: list[SamplingParams] | SamplingParams | None = None,
    ):
        if generate and sampling_params is not None:
            params = (
                sampling_params
                if isinstance(sampling_params, list)
                else [sampling_params] * len(requests)
            )
            # A caller may reuse one SamplingParams object for several requests
            # or batches. Copy each occurrence before assigning its own seed.
            sampling_params = [deepcopy(param) for param in params]
            for offset, params_for_request in enumerate(sampling_params):
                base_seed = (
                    params_for_request.seed
                    if params_for_request.seed is not None
                    else self.model_args["seed"]
                )
                params_for_request.seed = (
                    int(base_seed) + self._sampling_request_index + offset
                )
            self._sampling_request_index += len(sampling_params)

        outputs = super()._model_generate(
            requests=requests,
            generate=generate,
            sampling_params=sampling_params,
        )
        if generate:
            for output in outputs:
                for completion in output.outputs:
                    completion.text = _GeneratedText(
                        completion.text,
                        generated_token_count=len(completion.token_ids),
                    )
        return outputs
