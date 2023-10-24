from typing import Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50

try:
    from transformers import DistilBertConfig, DistilBertModel
except ImportError:
    raise ImportError(
        "`transformers` package not found, please run `pip install transformers`"
    )


def load_vision_model(model_name: Literal["resnet50"]) -> nn.Module:
    """
    Load a vision encoder model.
    Params:
        model_name(str): name of the model to load.
    """
    match model_name.lower():
        case "resnet50":
            model = resnet50()
        case _:
            raise NotImplementedError(f"Model name {model_name} is not implemented.")

    return model


def load_text_model(model_name: Literal["distilbert"]) -> nn.Module:
    """
    Load a text encoder model.
    Params:
        model_name(str): name of the model to load.
    """
    match model_name.lower():
        case "distilbert":
            bert_config = DistilBertConfig()
            model = DistilBertModel(bert_config)
        case _:
            raise NotImplementedError(f"Model name {model_name} is not implemented.")

    return model


class ClipModel(nn.Module):
    def __init__(
        self,
        vision: Literal["resnet50"],
        text: Literal["distilbert"],
        avg_word_embs: bool,
    ):
        """
        Simple clip model using HF transformers and torchvision models.
        """
        self.vision = load_vision_model(vision)
        self.transformer = load_text_model(text)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.avg_word_embs = avg_word_embs

    def encode_image(self, image: Tensor) -> Tensor:
        return self.vision(image)

    def encode_text(self, inputs: Dict[str, Tensor]) -> Tensor:
        if self.avg_word_embs:
            sequence_output = self.transformer(**inputs).last_hidden_state
            embeddings = torch.sum(
                sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
            ) / torch.clamp(
                torch.sum(inputs["attention_mask"], dim=1, keepdim=True), min=1e-9
            )

            return embeddings
        else:
            return self.transformer(**inputs).last_hidden_state[:, 0]

    def forward(
        self, image: Tensor, attention_mask: Tensor, input_ids: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        text = {"attention_mask": attention_mask, "input_ids": input_ids}
        image_embeds = self.encode_image(image)
        text_embeds = self.encode_text(text)
        return image_embeds, text_embeds, self.logit_scale.exp()
