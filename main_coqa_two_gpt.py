import os
import time
import numpy as np
import pandas as pd
import tqdm
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, set_seed

from torchfly.training.arguments import BaseArguments

import os, sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

from utils import *


args = BaseArguments() 
args.add_argument("--model_size",
                  type=str,
                  help="Model size")

args.add_argument("--dataset_name",
                  type=str,
                  help="dataset to use")

args.add_argument("--random_seed",
                  type=int,
                  help="random_seed")

args.add_argument("--use_all_loss",
                          action='store_true',
                          help="Use base model or not")
args.add_argument("--loss_discount",
                  type=float,
                  default=1.0,
                  help="the loss discount for turn before the final two turn, work when use_all_loss=True")

args = args.parse_args()

set_seed(args.random_seed)

# init all datasets
[train_dataset_dir, val_dataset_dir, test_dataset_dir] = get_dataset_dir_by_name(args.dataset_name)
train_dataset_dir = "../" + train_dataset_dir
val_dataset_dir = "../" + val_dataset_dir
test_dataset_dir = "../" + test_dataset_dir

train_data = torch.load(train_dataset_dir)
#train_data = train_data[:20]
val_data = torch.load(val_dataset_dir)
#val_data = val_data[:20]
test_data = torch.load(test_dataset_dir)


class TwoGPTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        # cancel it
        # self.tokenizer.max_len = 1500
        # tokenizer weird behavior
        self.turn_ending = [628, 198]
        
        self.__process__()
        # tokenizer.encode("\n\n\n") return [], but tokenizer.decode([628, 198]) return "\n\n\n"
        
    def __len__(self):
        return len(self.data)
    
    def __process__(self):
        self.processed_data = []
        for one in self.data:
            one_dial_tokens = [tokenizer.encode(item) + self.turn_ending for item in one]
            one_role_ids = [idx%2 for idx in range(len(one_dial_tokens))]
            self.processed_data.append([one_role_ids, one_dial_tokens])
            
    def __getitem__(self, index):
        dial_tokens = [tokenizer.encode(item) + self.turn_ending for item in self.data[index]]
        # since we use two GPT, it maybe not necessary to use role_id
        # role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
        # return role_ids, dial_tokens
        #role_ids = [idx%2 for idx in range(len(dial_tokens))]
        [role_ids, dial_tokens] = self.processed_data[index]
        return role_ids, dial_tokens
        
    def collate(self, unpacked_data):
        return unpacked_data

# load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
train_dataset = TwoGPTDataset(train_data, tokenizer)
val_dataset = TwoGPTDataset(val_data, tokenizer)
test_dataset = TwoGPTDataset(test_data, tokenizer)

train_dataloader = DataLoader(dataset=train_dataset, 
                              shuffle=True, 
                              batch_size=args.batch_size, 
                              collate_fn=train_dataset.collate)

val_dataloader = DataLoader(dataset=val_dataset, 
                            shuffle=False, 
                            batch_size=args.batch_size, 
                            collate_fn=train_dataset.collate)

test_dataloader = DataLoader(dataset=test_dataset, 
                            shuffle=False, 
                            batch_size=args.batch_size, 
                            collate_fn=train_dataset.collate)


# load the model
if args.model_size == "small":
    model_type = "gpt2" 
elif args.model_size == "medium":
    model_type = "gpt2-medium"
else:
    raise NotImplementedError()

model_A = GPT2LMHeadModel.from_pretrained(model_type)
model_B = GPT2LMHeadModel.from_pretrained(model_type)

device = torch.device("cuda")
model_A = model_A.to(device)
model_B = model_B.to(device)
# model_B = model_A


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None, loss_discount=1, end_length=-1):
        """
        reduce: None, "batch", "sentence"
        """
        return sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce, loss_discount, end_length)


def sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce, loss_discount=1.0, end_length=-1):
    # type: (Tensor, Tensor, Tensor, float, bool)-> Tensor
    """
    label_smoothing : ``float``, optional (default = 0.0)
        It should be smaller than 1.
    """

    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
                                       
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])
    
    # shape : (batch, sequence_length)
    # batch size = 1
    # mask[0][:-20] = 0
    loss = negative_log_likelihood * mask
    # loss = loss.unsqueeze(0)
    # breakpoint()

    if reduce:
        # shape : (batch,)
        # breakpoint()
        #assert len(loss) == 1

        # only use in training time
        if end_length != -1 and reduce == "batch":
            
            # breakpoint()
            loss[:, :-end_length] = loss[:, :-end_length] * loss_discount
        
        loss = loss.sum(1) / (mask.sum(1) + 1e-13)
        
        if reduce == "batch":
            # shape : scalar
            loss = loss.mean()

    return loss


def train_one_iter(batch, update_count, fp16=False):
    role_ids, dialog_tokens = batch
    dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]
    
    past = None
    all_logits = []
    
    for turn_num, dial_turn_inputs in enumerate(dial_inputs):
        
        if role_ids[turn_num] == 0:
            # breakpoint()
            logits, past = model_A(dial_turn_inputs, past=past)

            if args.use_all_loss:
                all_logits.append(logits)
            elif turn_num == len(dial_inputs) - 1 or turn_num == len(dial_inputs) - 2:
                all_logits.append(logits)
        else:
            # breakpoint()
            logits, past = model_B(dial_turn_inputs, past=past)
            if args.use_all_loss:
                all_logits.append(logits)
            elif turn_num == len(dial_inputs) - 1 or turn_num == len(dial_inputs) - 2:
                all_logits.append(logits)

    # breakpoint()
    length = all_logits[-2].shape[1] + all_logits[-1].shape[1] - 1
    all_logits = torch.cat(all_logits, dim=1)
    
    # target
    all_logits = all_logits[:, :-1].contiguous()
    target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
    target_mask = torch.ones_like(target).float()
    # breakpoint()
    
    # only last two turn loss
    if args.use_all_loss:
        loss = criterion(all_logits, target[:, :], target_mask[:, :],
                         label_smoothing=0.02, reduce="batch",
                         loss_discount=args.loss_discount, end_length=length)
    else:
        # length = all_logits.shape[1]
        loss = criterion(all_logits, target[:, -length:], target_mask[:, -length:], 
                            label_smoothing=0.02, reduce="batch") 

    loss /= args.gradient_accumulation_steps

    loss.backward()
        
    record_loss = loss.item() * args.gradient_accumulation_steps
    perplexity = np.exp(record_loss)
    
    return record_loss, perplexity


def validate(dataloader):
    with torch.no_grad():
        pbar = progress_bar(dataloader)

        total_ppl = []

        for batch in pbar:
            
            if sum([len(item) for item in batch[0][1]]) > 1024:
                continue
            
            role_ids, dialog_tokens = batch[0]
            dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]

            past = None
            all_logits = []

            for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                if role_ids[turn_num] == 0:
                    logits, past = model_A(dial_turn_inputs, past=past)
                    # all_logits.append(logits)
                    if turn_num == len(dial_inputs) - 2:
                        all_logits.append(logits)
                else:
                    logits, past = model_B(dial_turn_inputs, past=past)
                    if turn_num == len(dial_inputs) - 1:
                        all_logits.append(logits)
                        length_last_question = logits.shape[1]
            
            assert len(all_logits) == 2
            all_logits = torch.cat(all_logits, dim=1)
            
            # target
            all_logits = all_logits[:, :-1].contiguous()
            target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
            target_mask = torch.ones_like(target).float()
            
            loss = criterion(all_logits[:, -length_last_question:, :], target[:, -length_last_question:], \
                             target_mask[:, -length_last_question:], label_smoothing=-1, reduce="sentence")      
            # breakpoint()
            ppl = torch.exp(loss)
            total_ppl.extend(ppl.tolist())

        print(f"Epcoh {ep} Validation Perplexity: {np.mean(total_ppl)} Variance: {np.var(total_ppl)}")
        
        return np.mean(total_ppl)

criterion = SequenceCrossEntropyLoss()

# make Checkpoint dir
if args.model_size == "small":
    _checkpoint_dir = args.dataset_name + "_small_Checkpoint"
elif args.model_size == "medium":
    _checkpoint_dir = args.dataset_name + "_medium_Checkpoint"
elif args.model_size == "large":
    _checkpoint_dir = args.dataset_name + "_large_Checkpoint"


for i in range(1, 10):
    temp = _checkpoint_dir + "_" + str(i)
    if not os.path.isdir(temp):
        args.checkpoint_dir = temp
        break

os.makedirs(args.checkpoint_dir, exist_ok=False)

print(dict_to_text(args.__dict__))

# store config for each checkpoint folder
config_loc = args.checkpoint_dir + "/config.json"
config = copy.deepcopy(args.__dict__)

with open(config_loc, "w") as f:
    json.dump(config, f, indent=4)

# define hyper-parameters
num_train_optimization_steps = num_train_optimization_steps = len(train_dataset) * args.num_train_epochs // args.batch_size // args.gradient_accumulation_steps

param_optimizer = list(model_A.named_parameters()) + list(model_B.named_parameters())
no_decay = ['bias', 'ln', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


optimizer = AdamW(optimizer_grouped_parameters, 
                  lr=args.learning_rate,
                  eps=1e-06)

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps,
                                            num_training_steps=num_train_optimization_steps)


os.makedirs("models", exist_ok=True)


update_count = 0
progress_bar = tqdm.tqdm
start = time.time()
old_ppl = -float('Inf')

for ep in range(args.num_train_epochs):

    "Training"
    pbar = progress_bar(train_dataloader)
    model_A.train()
    model_B.train()
    
    for batch in pbar:
        batch = batch[0]
        
        # without relative position, we skip dialogs
        #
        # if sum([len(item) for item in batch[1]]) > 1024:
        #   continue
        
        if sum([len(item) for item in batch[1]]) > 1024:
            continue
            
        record_loss, perplexity = train_one_iter(batch, update_count, fp16=args.fp16)
        
        update_count += 1

        if update_count % args.gradient_accumulation_steps == args.gradient_accumulation_steps - 1:
            # update for gradient accumulation
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # speed measure
            end = time.time()
            speed = args.batch_size * args.gradient_accumulation_steps / (end - start)
            start = end
            
            # show progress
            pbar.set_postfix(loss=record_loss, perplexity=perplexity, speed=speed)

    "Evaluation"
    model_A.eval()
    model_B.eval()
    ppl = validate(val_dataloader)
    
    # save the model for later use
    filepath = os.path.join(args.checkpoint_dir, f"model_iter_{update_count}.pth")
    torch.save([model_A.state_dict(), model_B.state_dict()], filepath)
    print("saving at {}".format(filepath))