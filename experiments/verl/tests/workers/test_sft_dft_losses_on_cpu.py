# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import unittest

import torch
from omegaconf import OmegaConf

from verl.trainer.sft_trainer import SFTTrainer, _mean_metric_value, _resolve_loss_function
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.losses import dft_loss, sft_loss


class TestSFTDFTLosses(unittest.TestCase):
    def test_validation_drops_only_incomplete_final_batch(self):
        class FakeEngine:
            @staticmethod
            def get_data_parallel_rank():
                return 0

            @staticmethod
            def get_data_parallel_size():
                return 2

        trainer = object.__new__(SFTTrainer)
        trainer.config = OmegaConf.create(
            {
                "data": {
                    "train_batch_size": 256,
                    "pad_mode": "right",
                    "num_workers": 0,
                    "val_drop_last": True,
                },
                "trainer": {"seed": 1},
            }
        )
        trainer.engine = FakeEngine()
        trainer.train_dataset = list(range(1_000))
        trainer.val_dataset = list(range(600))

        trainer._build_dataloader()

        self.assertTrue(trainer.train_dataloader.drop_last)
        self.assertTrue(trainer.val_dataloader.drop_last)
        self.assertEqual(len(trainer.val_sampler), 300)
        self.assertEqual(len(trainer.val_dataloader), 2)
        self.assertEqual(
            sum(len(indices) for indices in trainer.val_dataloader.batch_sampler) * 2,
            512,
        )

    def test_sft_loss_matches_masked_nll_with_padding(self):
        log_prob = torch.tensor([[-0.2, -1.0, -2.0], [-0.5, -0.7, -1.2]])
        response_mask = torch.tensor([[True, False, True], [False, True, False]])
        data = {
            "pad_mode": DatasetPadMode.RIGHT,
            "response_mask": response_mask,
            "batch_num_tokens": 3,
            "dp_size": 2,
        }

        loss, metrics = sft_loss(config=None, model_output={"log_probs": log_prob}, data=data)

        expected = -(log_prob * response_mask).sum() / 3 * 2
        torch.testing.assert_close(loss, expected)
        self.assertEqual(metrics, {})

    def test_dft_loss_uses_detached_token_probability_with_padding(self):
        log_prob = torch.tensor([[-0.2, -1.0, -2.0], [-0.5, -0.7, -1.2]], requires_grad=True)
        response_mask = torch.tensor([[True, False, True], [False, True, False]])
        data = {
            "pad_mode": DatasetPadMode.RIGHT,
            "response_mask": response_mask,
            "batch_num_tokens": 3,
            "dp_size": 2,
        }

        loss, metrics = dft_loss(config=None, model_output={"log_probs": log_prob}, data=data)
        probability = log_prob.detach().exp()
        expected = -(log_prob * probability * response_mask).sum() / 3 * 2
        original_loss = -(log_prob.detach() * response_mask).sum() / 3 * 2

        torch.testing.assert_close(loss, expected)
        self.assertAlmostEqual(metrics["original_loss"].aggregate(), original_loss.item())

        loss.backward()
        expected_grad = -probability * response_mask / 3 * 2
        torch.testing.assert_close(log_prob.grad, expected_grad)

    def test_dft_loss_uses_shifted_loss_mask_without_padding(self):
        log_prob_values = torch.tensor([-0.2, -1.0, -2.0, -0.5, -0.7])
        offsets = torch.tensor([0, 3, 5])
        log_prob = torch.nested.nested_tensor_from_jagged(log_prob_values, offsets)
        loss_mask_values = torch.tensor([0, 1, 1, 0, 1])
        loss_mask = torch.nested.nested_tensor_from_jagged(loss_mask_values, offsets)
        shifted_loss_mask = torch.roll(loss_mask_values, shifts=-1)
        data = {
            "pad_mode": DatasetPadMode.NO_PADDING,
            "loss_mask": loss_mask,
            "batch_num_tokens": int(shifted_loss_mask.sum()),
            "dp_size": 3,
        }

        loss, metrics = dft_loss(config=None, model_output={"log_probs": log_prob}, data=data)

        probability = log_prob_values.exp()
        expected = -(log_prob_values * probability * shifted_loss_mask).sum() / shifted_loss_mask.sum() * 3
        original_loss = -(log_prob_values * shifted_loss_mask).sum() / shifted_loss_mask.sum() * 3
        torch.testing.assert_close(loss, expected)
        self.assertAlmostEqual(metrics["original_loss"].aggregate(), original_loss.item())

        sft_value, _ = sft_loss(config=None, model_output={"log_probs": log_prob}, data=data)
        torch.testing.assert_close(sft_value, original_loss)

    def test_loss_type_resolution_and_distributed_sum_metric(self):
        self.assertIs(_resolve_loss_function("sft"), sft_loss)
        self.assertIs(_resolve_loss_function("dft"), dft_loss)
        with self.assertRaisesRegex(ValueError, "Unsupported trainer.loss_type='other'"):
            _resolve_loss_function("other")

        rank_0 = Metric(value=1.0, aggregation=AggregationType.SUM)
        rank_0.append(2.0)
        rank_1 = Metric(value=3.0, aggregation=AggregationType.SUM)
        rank_1.append(4.0)
        self.assertAlmostEqual(_mean_metric_value([rank_0, rank_1]), 5.0)


if __name__ == "__main__":
    unittest.main()
