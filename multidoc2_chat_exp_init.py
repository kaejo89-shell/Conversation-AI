import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import json
from transformers import RobertaTokenizerFast
from transformers import EncoderDecoderModel
from torch.utils.data import Dataset
from dataclasses import dataclass
import torch
from typing import Optional, Union, Callable, Dict, List, Tuple
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    DataCollator,
    PreTrainedTokenizerBase,
    EvalPrediction,
    TrainerCallback,
)
import torch.nn as nn
import numpy as np
from functools import partial
from transformers.trainer_callback import EarlyStoppingCallback

boolean = bool
encoder_model_base = "roberta-base"
decoder_model_base = "roberta-base"

@dataclass
class Features:
    input_ids: List[int]
    attention_mask: List[int]
    labels: Optional[List[int]]
    decoder_attention_mask: Optional[List[int]]


class MultiDoc2ChatDataset(Dataset):
    def __init__(self, tokenizer, data_pack) -> None:

        self.tokenizer = tokenizer
        self.data_pack = data_pack
        self.change_data_mode(1)
        super().__init__()

    def change_data_mode(self, mode=1):
        self.mode = mode > 1

    def __len__(
        self,
    ):
        return len(self.data_pack)

    def processExample(self, data):

        facts = data["facts"]
        utterance = data["utterance"]
        question = data["current_question"]
        conv_history = data["conv_history"].replace("||", " [SEP] ")

        if len(conv_history) == 0:
            conv_history = "not available"

        fact_question_passage = "[CONV_HISTORY] " + conv_history + \
            " [CURRENT_QUESTION] "+question + " [FACTS] " + facts
        response_passage = "[RESPONSE] " + utterance

        # apply the tokenizer to convert the texts to the appropriate input
        if not self.mode:
            label_pack = self.tokenizer(response_passage, return_tensors="pt")
            label_seq = label_pack["input_ids"].flatten()
            label_attention = label_pack["attention_mask"].flatten()

        passage_pack = self.tokenizer(
            fact_question_passage, return_tensors="pt")

        passage_seq = passage_pack["input_ids"].flatten()
        passage_attention = passage_pack["attention_mask"].flatten()

        if not self.mode:
            return Features(
                input_ids=passage_seq,
                attention_mask=passage_attention,
                labels=label_seq,
                decoder_attention_mask=label_attention,
            )
        else:
            return Features(
                input_ids=passage_seq,
                attention_mask=passage_attention,
                labels=[],
                decoder_attention_mask=[],
            )

    def __getitem__(self, index):
        return self.processExample(self.data_pack[index])


def pad_seq(
    seq: List[np.ndarray], max_batch_len: int, pad_value: int, verbose=False
) -> List[int]:
    if len(seq) > max_batch_len:
        seq = seq.to(torch.long).unsqueeze(0)[:, :max_batch_len]
        return seq
    pads = torch.from_numpy(np.array([pad_value] * (max_batch_len - len(seq))))
    out = torch.concat([seq, pads], -1).to(torch.long).unsqueeze(0)
    return out


@dataclass
class SmartCollator:
    pad_token_id: int
    label_pad_token_id: int = -100
    is_gpt: boolean = False
    max_input_len: int = 400
    max_output_len: int = 250
    is_inference: boolean = False

    def __call__(self, batch: List[Features]) -> Dict[str, torch.Tensor]:
        batch_inputs: List = list()
        batch_attention_masks: List = list()
        decoder_attention_mask: List = list()
        labels: List = list()
        max_size = min([max([len(ex.input_ids)
                       for ex in batch]), self.max_input_len])

        max_size_output = min(
            [max([len(ex.labels) for ex in batch]), self.max_output_len]
        )  # type: ignore

        for item in batch:
            batch_inputs += [pad_seq(item.input_ids,
                                     max_size, self.pad_token_id)]
            batch_attention_masks += [
                pad_seq(item.attention_mask, max_size, 0)]

            if not self.is_gpt and not self.is_inference:
                decoder_attention_mask += [
                    pad_seq(item.decoder_attention_mask, max_size_output, 0)
                ]
            if not self.is_inference:
                labels += [
                    pad_seq(item.labels, max_size_output,
                            self.label_pad_token_id)
                ]
        if not self.is_gpt:
            if not self.is_inference:
                return dict(
                    input_ids=torch.concat(batch_inputs, 0),
                    attention_mask=torch.concat(batch_attention_masks, 0),
                    labels=torch.concat(labels, 0),
                    decoder_attention_mask=torch.concat(
                        decoder_attention_mask, 0),
                )
            else:
                return dict(
                    input_ids=torch.concat(batch_inputs, 0),
                    attention_mask=torch.concat(batch_attention_masks, 0),
                )
        else:
            return dict(
                input_ids=torch.concat(batch_inputs, 0),
                attention_mask=torch.concat(batch_attention_masks, 0),
                labels=torch.concat(labels, 0),
            )


@dataclass
class RunArguments:

    output_dir: str

    learning_rate: float = 5e-5
    run_id: str = ""
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    max_seq_len: int = 600
    warmup_ratio: float = 0.25
    eval_steps: int = 1000
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    lr_scheduler_type: str = "cosine"
    save_total_limit: int = 1
    num_train_epochs: int = 5
    weight_decay: float = 0.3
    verbose: bool = False


def model_init(
    device=torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    generator = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_model_base,
        decoder_model_base,
    )
    # update the tokens
    generator.encoder.resize_token_embeddings(len(tokenizer))
    generator.decoder.resize_token_embeddings(len(tokenizer))

    # set the decoder start and end tokens
    # generator.config.decoder_start_token_id = tokenizer.bos_token_id
    generator.config.decoder_start_token_id = tokenizer.bos_token_id
    generator.config.eos_token_id = tokenizer.eos_token_id
    generator.config.vocab_size = generator.config.encoder.vocab_size
    generator.config.pad_token_id = tokenizer.pad_token_id
    print(f"\n\nNum Params: {generator.num_parameters()}")

    return generator.to(device)


def get_model_trainer_arguments(args):
    return TrainingArguments(
        overwrite_output_dir=True,
        adafactor=False,
        load_best_model_at_end=True,
        output_dir=args.output_dir + "/" + args.run_id + "/",
        evaluation_strategy=args.evaluation_strategy,  # "epoch",
        save_strategy=args.save_strategy,  # 'epoch',
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        save_total_limit=args.save_total_limit,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        disable_tqdm=not args.verbose,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
    )


class CustomTrainer(Trainer):
    def __init__(
        self,
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        model: Union[PreTrainedModel, nn.Module] = None,  # type: ignore
        args: TrainingArguments = None,  # type: ignore
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = None,  # type: ignore
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.device = device

    def compute_loss(self, model, batch, return_outputs=False):

        b_input_ids = batch["input_ids"].to(self.device)
        b_input_mask = batch["attention_mask"].to(self.device)
        b_labels = batch["labels"].to(self.device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(
            self.device)

        outputs = model(
            b_input_ids,
            attention_mask=b_input_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=b_labels,
        )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss



tokenizer = RobertaTokenizerFast.from_pretrained(encoder_model_base)
special_tokens = ["[FACTS]", "[RESPONSE]", "[CONV_HISTORY]","[CURRENT_QUESTION]"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})


# Load the data
with open("processed_data/test_data.json", "r") as test_file:
    test_data = json.load(test_file)

with open("processed_data/dev_data.json", "r") as dev_file:
    dev_data = json.load(dev_file)

with open("processed_data/train_data.json", "r") as train_file:
    train_data = json.load(train_file)

train_dataset = MultiDoc2ChatDataset(tokenizer, train_data)
test_dataset = MultiDoc2ChatDataset(tokenizer, test_data)
dev_dataset = MultiDoc2ChatDataset(tokenizer, dev_data)