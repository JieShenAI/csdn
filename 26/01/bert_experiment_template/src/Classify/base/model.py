import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel


class DiyConfig(PretrainedConfig):
    def __init__(self, hf_name=None, num_label=-1, **kwargs):
        super().__init__(**kwargs)
        self.num_label = num_label
        self.hf_name = hf_name


class DiyModel(PreTrainedModel):
    config_class = DiyConfig

    def __init__(self, config):
        super().__init__(config)
        print("custom config")
        print(config)
        self.model = AutoModel.from_pretrained(config.hf_name)
        print(self.model.config)

        self.hidden_size: int = self.model.config.hidden_size
        self.config = config
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.block = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.linear = nn.Linear(self.hidden_size, config.num_label)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, text_tokens, labels=None):
        output = self.model(**text_tokens)
        cls_tensor = output.last_hidden_state[:, 0]
        logits = cls_tensor + self.block(self.layer_norm(cls_tensor))
        logits = self.linear(logits)
        if labels is not None:
            return {
                "logits": logits,
                "loss": self.ce_loss(logits, labels),
            }
        return {
            "logits": logits
        }
