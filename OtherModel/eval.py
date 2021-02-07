import logging
from dataclasses import dataclass, field
from typing import Optional

import os
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser

from data_collator import T2TDataCollator
import argparse
import pickle

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    BartTokenizer,
    HfArgumentParser,
    DataCollator,
    TrainingArguments,
    set_seed,
)



class BaseArguments:
    """
    The most commonly used arguments
    """
    def __init__(self):
        self._parser = argparse.ArgumentParser(description="")

        self._parser.add_argument(
            "--config_file",
            default=None,
            type=str,
            help="config file location")

    def add_argument(self, *args, **kwargs):
        self._parser.add_argument(*args, **kwargs)

    def parse_args(self, arguments=None):
        is_notebook =  False #check_if_notebook()

        if is_notebook:
            sys.argv = ['']

        if arguments:
            args = self._parser.parse_args(arguments)
        else:
            args = self._parser.parse_args()

        args.is_notebook = is_notebook

        return args


MODEL_TYPE_TO_TOKENIZER = {
    "t5": T5Tokenizer,
    "bart": BartTokenizer,
}

logger = logging.getLogger(__name__)

@dataclass
class EvalArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    eval_file_path: str = field(
        metadata={"help": "Path for cached dev dataset"}
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    num_beams: Optional[int] = field(
        default=4,
        metadata={"help": "num_beams to use for decoding"}
    )
    max_decoding_length: Optional[int] = field(
        default=32,
        metadata={"help": "maximum length for decoding"}
    )
    output_path: Optional[str] = field(
        default="hypothesis.txt",
        metadata={"help": "path to save the generated questions."}
    )
 
def get_predictions(model, tokenizer, data_loader, device="cpu", num_beams=4, max_length=32, length_penalty=1):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            #breakpoint()
            outs = model.generate(
                input_ids=batch['input_ids'].to(device), 
                attention_mask=batch['attention_mask'].to(device),
                num_beams=num_beams,
                max_length=max_length,
                length_penalty=length_penalty,
            )

            prediction = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            label_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["labels"]]
            context = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["input_ids"]]

            # breakpoint()
            
            for a, b, c in zip(context, label_text, prediction):
                predictions.append({"context":a, "label":b, "prediction":c})

            #return predictions
    return predictions

def main():

    args_json = BaseArguments() 
    args_json = args_json.parse_args()

    parser = HfArgumentParser((EvalArguments,TrainingArguments))
    args, training_args = parser.parse_json_file(json_file=args_json.config_file)

    if args.model_type=="bart":
        prefix = "outputs/bart-large/checkpoint-"
        times = 10
        per_count = 1000
    elif args.model_type=="t5":
        prefix = "outputs/t5-large/checkpoint-"
        times = 5
        per_count = 500
    else:
        raise ValueError()

    for i in range(times):
        count = (i + 1) * per_count
        args.model_name_or_path = prefix + str(count)

    # breakpoint()

        tokenizer_cls = MODEL_TYPE_TO_TOKENIZER[args.model_type]
        tokenizer = tokenizer_cls.from_pretrained(
            args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_batchname_or_path,
        )

        # breakpoint()

        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
        device = training_args.device
        model.to(device)

        eval_dataset = torch.load("data/" + args.eval_file_path)
        collator = T2TDataCollator(
            tokenizer=tokenizer,
            model_type=args.model_type,
            mode="inference"
        )
        loader = torch.utils.data.DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=collator)

        predictions = get_predictions(
            model=model,
            tokenizer=tokenizer,
            data_loader=loader,
            device=device,
            num_beams=args.num_beams,
            max_length=args.max_decoding_length
        )

        #breakpoint()
        args.output_path = os.path.join(args.model_name_or_path, "generated_text") 
        with open(args.output_path, 'wb') as f:
            pickle.dump(predictions, f)
        
        logging.info(f"Output saved at {args.output_path}")


if __name__ == "__main__":
    main()
