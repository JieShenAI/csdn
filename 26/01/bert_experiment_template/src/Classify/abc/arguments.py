from dataclasses import dataclass, field
from transformers import AutoConfig, AutoTokenizer, TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    hf_name: str = field(
        metadata={"help": "Pre-trained model name in huggingface"}
    )
    num_label: int = field(default=-1, metadata={"help": "Number of labels"})


@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "dataset in huggingface or local dataset directory"})
    text_max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
