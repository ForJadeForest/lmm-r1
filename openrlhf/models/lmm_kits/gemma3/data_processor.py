from typing import List, Dict, Union
import json

import torch

from ..base.data_processor import BaseDataProcessor, MMInputs


class Gemma3_VLDataProcessor(BaseDataProcessor):
    def __call__(
        self,
        messages,
        max_length,
        padding=True,
        device=None,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
    ) -> MMInputs:
        messages = self._format_messages(messages)
        processor = self.processor
        batch = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=padding,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_tensors=return_tensors,
            return_dict=True,
        )
        emb_inputs, extra_info, forward_inputs = self._split_input_dict(batch)
        return MMInputs(emb_inputs=emb_inputs, extra_info=extra_info, forward_inputs=forward_inputs).to(device)

    def _split_input_dict(self, input_dict: Dict) -> tuple[Dict, Dict]:
        extra_info = {}
        forward_inputs = {}
        if "input_ids" in input_dict:
            extra_info["input_ids"] = input_dict.pop("input_ids")
        if "attention_mask" in input_dict:
            extra_info["attention_mask"] = input_dict.pop("attention_mask")
        if "token_type_ids" in input_dict:
            forward_inputs["token_type_ids"] = input_dict.pop("token_type_ids")
        return input_dict, extra_info, forward_inputs

    def make_input_batch(self, inputs: List[MMInputs]) -> MMInputs:
        # each element has no batch dimension
        batch = {}
        # collect all keys
        for inp in inputs:
            batch.update({k: None for k, v in inp.items() if v is not None})
        for k in batch.keys():
            if k in ["input_ids", "attention_mask", "token_type_ids"]:
                batch[k] = torch.stack([inp[k] for inp in inputs if k in inp], dim=0)
            elif k in ["pixel_values"]:
                # concat all patches of all images in a batch in the first dimension
                batch[k] = torch.cat([inp[k] for inp in inputs if k in inp], dim=0)
            else:
                raise ValueError(f"Unknown key {k} for Gemma3_VLDataProcessor")
        emb_inputs, extra_info, forward_inputs = self._split_input_dict(batch)
        return MMInputs(emb_inputs=emb_inputs, extra_info=extra_info, forward_inputs=forward_inputs)

    def split_input_batch(self, batch: MMInputs) -> List[MMInputs]:
        batch_size = len(batch["input_ids"])
        batch_kwargs = [{} for _ in range(batch_size)]
        # first process None values
        keys = []
        for k, v in batch.items():
            if v is not None:
                keys.append(k)
            else:
                for i in range(batch_size):
                    batch_kwargs[i][k] = None

        if "pixel_values" in keys and ("input_ids" not in keys):
            raise ValueError("Cannot split batch with pixel_values without input_ids")

        for k in ["input_ids", "attention_mask", "token_type_ids"]:
            if k in keys:
                vals = batch[k]
                if isinstance(vals, torch.Tensor):
                    vals = torch.unbind(vals)
                assert batch_size == len(vals)
                for i, v in enumerate(vals):
                    batch_kwargs[i][k] = v
        if "pixel_values" in keys:
            pixel_values = batch["pixel_values"]
            for i in range(batch_size):
                token_type_ids_i = batch_kwargs[i]["token_type_ids"]
                assert (token_type_ids_i == 1).sum() % 256 == 0
                img_num = (token_type_ids_i == 1).sum().item() // 256
                if img_num == 0:
                    batch_kwargs[i]["pixel_values"] = None
                    continue

                pixel_values_i = pixel_values[:img_num]
                assert len(pixel_values_i) == img_num
                pixel_values = pixel_values[img_num:]
                batch_kwargs[i]["pixel_values"] = pixel_values_i
            assert len(pixel_values) == 0, f"{pixel_values.shape}"
        mm_inputs_list = []
        for b in batch_kwargs:
            emb_inputs, extra_info, forward_inputs = self._split_input_dict(b)
            mm_inputs_list.append(
                MMInputs(emb_inputs=emb_inputs, extra_info=extra_info, forward_inputs=forward_inputs)
            )
        return mm_inputs_list

    def warp_str_content_to_dict(self, messages_list: List[List[Dict]]):
        """
        Gemma Processor needs the content key to be a list of dict.
        """
        for messages in messages_list:
            for message in messages:
                if isinstance(message["content"], str):
                    message["content"] = [{"type": "text", "text": message["content"]}]
                else:
                    assert isinstance(message["content"], list)
        return messages_list

    def _format_messages(self, messages: Union[Dict, List[str], str]) -> List[List[Dict]]:
        formated_messages = super()._format_messages(messages)
        formated_messages = self.warp_str_content_to_dict(formated_messages)

        return formated_messages


DataProcessor = Gemma3_VLDataProcessor

__all__ = ["DataProcessor"]
