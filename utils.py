import numpy as np
import math
import torch
from torch.autograd import grad


def buffer_loader(buffer, mini_batch_size, shuffle=False):
    """
    A mini batch iterator
    """
    indices = np.arange(len(buffer))
    if shuffle:
        np.random.shuffle(indices)

    for i in range(math.ceil(len(buffer) / mini_batch_size)):
        sampled_indices = indices[i * mini_batch_size:(i + 1) *
                                  mini_batch_size]
        # get sampled batch
        yield sampled_indices, [buffer[idx] for idx in sampled_indices]


def get_optim_weights(model, weight_decay=0.01):
    """No weight decay for bias and LayerNorm
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ['ln', 'bias', 'LayerNorm']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        weight_decay
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    return optimizer_grouped_parameters


def dict_to_text(inputs: dict):
    return "{" + "\n".join("{!r}: {!r},".format(k, v)
                           for k, v in inputs.items()) + "}"



def get_dataset_dir_by_name(dataset_name):

    if dataset_name == "persuasion":
        train_dataset_dir = "data/persuasion/train_persuasion_leveled.pkl"
        val_dataset_dir = "data/persuasion/val_persuasion_leveled.pkl"
        # no test set for persuasion
        test_dataset_dir = "data/persuasion/val_persuasion_leveled.pkl"
    elif dataset_name == "dailydialog":
        train_dataset_dir = "data/ijcnlp_dailydialog/train_dailydialog_leveled.pkl"
        val_dataset_dir = "data/ijcnlp_dailydialog/val_dailydialog_leveled.pkl"
        test_dataset_dir = "data/ijcnlp_dailydialog/test_dailydialog_leveled.pkl"
    elif dataset_name == "personachat":
        train_dataset_dir = "data/personachat/train_personachat.pkl"
        val_dataset_dir = "data/personachat/val_personachat.pkl"
        test_dataset_dir = "data/personachat/test_personachat.pkl"
    elif dataset_name == "dstc-7":
        train_dataset_dir = "data/dstc7-task2/train_dstc.pkl"
        val_dataset_dir = "data/dstc7-task2/val_dstc.pkl"
        test_dataset_dir = "data/dstc7-task2/test_dstc.pkl"
    elif dataset_name == "dstc-7-sampled":
        train_dataset_dir = "data/dstc7-task2/sampled_train_dstc.pkl"
        val_dataset_dir = "data/dstc7-task2/val_dstc.pkl"
        test_dataset_dir = "data/dstc7-task2/test_dstc.pkl"
    elif dataset_name == "rocstory":
        train_dataset_dir = "data/rocstory/train_rocstory.pkl"
        val_dataset_dir = "data/rocstory/val_rocstory.pkl"
        test_dataset_dir = "data/rocstory/test_rocstory.pkl"
    elif dataset_name == "imdb_conditional":
        train_dataset_dir = "data/aclImdb_v1/train_imdb_conditional.pkl"
        val_dataset_dir = "data/aclImdb_v1/val_imdb_conditional.pkl"
        test_dataset_dir = "data/aclImdb_v1/test_imdb_conditional.pkl"
    elif dataset_name == "imdb_unconditional":
        train_dataset_dir = "data/aclImdb_v1/train_imdb_unconditional.pkl"
        val_dataset_dir = "data/aclImdb_v1/val_imdb_unconditional.pkl"
        test_dataset_dir = "data/aclImdb_v1/test_imdb_unconditional.pkl"
    elif dataset_name == "cocoCaption_unconditional":
        train_dataset_dir = "data/coco_caption/train_cocoCaption_unconditional.pkl"
        val_dataset_dir = "data/coco_caption/test_cocoCaption_unconditional.pkl"
        test_dataset_dir = "data/coco_caption/test_cocoCaption_unconditional.pkl"
    elif dataset_name == "squad":
        train_dataset_dir = "data/squad/train_squad1.pkl"
        val_dataset_dir = "data/squad/dev_squad1.pkl"
        test_dataset_dir = "data/squad/test_squad1.pkl"
    elif dataset_name == "coqa":
        train_dataset_dir = "data/coqa/train_coqa.pkl"
        val_dataset_dir = "data/coqa/dev_coqa.pkl"
        test_dataset_dir = "data/coqa/test_coqa.pkl"
    elif dataset_name == "coqa_no_history":
        train_dataset_dir = "data/coqa/train_coqa_no_history.pkl"
        val_dataset_dir = "data/coqa/dev_coqa_no_history.pkl"
        test_dataset_dir = "data/coqa/test_coqa_no_history.pkl"
    elif dataset_name == "coqa_QA_pair":
        train_dataset_dir = "data/coqa/train_coqa_QA_pair.pkl"
        val_dataset_dir = "data/coqa/dev_coqa_QA_pair.pkl"
        test_dataset_dir = "data/coqa/test_coqa_QA_pair.pkl" 
    elif dataset_name == "coqa_QA_order":
        train_dataset_dir = "data/coqa/train_coqa_QA_order.pkl"
        val_dataset_dir = "data/coqa/dev_coqa_QA_order.pkl"
        test_dataset_dir = "data/coqa/test_coqa_QA_order.pkl"
        
    elif dataset_name == "coqa_two_gpt":
        train_dataset_dir = "data/coqa/train_coqa_two_gpt.pkl"
        val_dataset_dir = "data/coqa/dev_coqa_two_gpt.pkl"
        test_dataset_dir = "data/coqa/test_coqa_two_gpt.pkl"
    
    elif dataset_name.startswith("coqa_two_gpt_mask"):
        train_dataset_dir = "data/coqa/train_" + dataset_name + ".pkl"
        val_dataset_dir = "data/coqa/dev_" + dataset_name + ".pkl"
        test_dataset_dir = "data/coqa/test_" + dataset_name + ".pkl"
    
    elif dataset_name.startswith("coqa_two_gpt_all_loss_"):
        train_dataset_dir = "data/coqa/train_coqa_two_gpt.pkl"
        val_dataset_dir = "data/coqa/dev_coqa_two_gpt.pkl"
        test_dataset_dir = "data/coqa/test_coqa_two_gpt.pkl"
        
    elif dataset_name == "coqa_two_gpt_no_history":
        train_dataset_dir = "data/coqa/train_coqa_two_gpt_no_history.pkl"
        val_dataset_dir = "data/coqa/dev_coqa_two_gpt_no_history.pkl"
        test_dataset_dir = "data/coqa/test_coqa_two_gpt_no_history.pkl"
    elif dataset_name == "coqa_two_gpt_no_highlight":
        train_dataset_dir = "data/coqa/train_coqa_two_gpt_no_highlight.pkl"
        val_dataset_dir = "data/coqa/dev_coqa_two_gpt_no_highlight.pkl"
        test_dataset_dir = "data/coqa/test_coqa_two_gpt_no_highlight.pkl" 
    elif dataset_name == "coqa_two_gpt_QA_order":
        train_dataset_dir = "data/coqa/train_coqa_two_gpt_QA_order.pkl"
        val_dataset_dir = "data/coqa/dev_coqa_two_gpt_QA_order.pkl"
        test_dataset_dir = "data/coqa/test_coqa_two_gpt_QA_order.pkl" 
    elif dataset_name == "quac":
        train_dataset_dir = "data/quac/train_quac.pkl"
        val_dataset_dir = "data/quac/dev_quac.pkl"
        test_dataset_dir = "data/quac/test_quac.pkl"
    elif dataset_name == "quac_no":
        train_dataset_dir = "data/quac/train_quac_no_cannotanswer.pkl"
        val_dataset_dir = "data/quac/dev_quac_no_cannotanswer.pkl"
        test_dataset_dir = "data/quac/test_quac_no_cannotanswer.pkl"    
    elif dataset_name == "quac_no_no":
        train_dataset_dir = "data/quac/train_quac_no_cannotanswer_no_type.pkl"
        val_dataset_dir = "data/quac/dev_quac_no_cannotanswer_no_type.pkl"
        test_dataset_dir = "data/quac/test_quac_no_cannotanswer_no_type.pkl"
    elif dataset_name == "quac_QA_pair_no":
        train_dataset_dir = "data/quac/train_quac_QA_pair_no_cannotanswer.pkl"
        val_dataset_dir = "data/quac/dev_quac_QA_pair_no_cannotanswer.pkl"
        test_dataset_dir = "data/quac/test_quac_QA_pair_no_cannotanswer.pkl"    



    else:
        raise NotImplementedError()
    
    return [train_dataset_dir, val_dataset_dir, test_dataset_dir]


def generate_result_store_dir(args, t, is_eval):

    # t is current drop out time index

    name = "generations/"
    name += args.dataset_name
    if "medium" in args.model_path:
        name += "_mediumG_"
    else:
        name += "_baseG_"

    if "unconditional" in args.model_path:
        name += "unconditional_"

    model_name = args.model_path.split("/")[-1].split(".")[0]
    
    # find checkpoint. Might use different hyperparameter to get different generation model of the same GPT size.
    full_checkpoint_name = args.model_path.split("/")[-2]
    checkpoint_name = full_checkpoint_name[full_checkpoint_name.find("Checkpoint"):]
    checkpoint_name = checkpoint_name + "_"

    name += checkpoint_name
    name += model_name
    if args.use_base:
        name += "_BaseMM_"
    else:
        name += "_LargeMM_"

    if not is_eval:
        name += "drop_"
        name += str(t)
    else:
        name += "nodrop"

    if args.use_train:
        name = name + "_train"
    else:
        name = name + "_val"

    temperature = "_T-" + str(args.test_temperature)

    name = name + temperature
    
    return name


def compute_gradient_penalty(input, output):
        # ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
        '''

        :param input: state[index]
        :param network: actor or critic
        :return: gradient penalty
        '''
        # input_ = torch.tensor(input).requires_grad_(True)
        # input_ = input.requires_grad_(True)
        # input_ = discriminator.collate_fn(input)
        # input_ = batch_to_device(input_, discriminator.args.device)

        # input_ = torch.stack((input_["input_ids"], input_["position_ids"], input_["input_mask"].long()))

        # output = network(input_)
        
        musk = torch.ones_like(output)

        gradients = grad(output, input, grad_outputs=musk,
                         retain_graph=True, create_graph=True,
                         allow_unused=True)[0]  # get tensor from tuple

        gradients = gradients.view(-1, 1)

        gradient_penalty = ((gradients.norm(2) - 1) ** 2).mean()
        return gradient_penalty
        


def batch_to_device(batch, device, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []

    # send batch to device
    if isinstance(batch, list):
        batch = [
            item.to(device, non_blocking=True) for item in batch
            if isinstance(item, torch.Tensor)
        ]
    if isinstance(batch, dict):
        batch = {
            k: v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            and all([key not in k for key in exclude_keys]) else v
            for k, v in batch.items()
        }
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device, non_blocking=True)
    else:
        raise NotImplementedError

    return batch

def dict_to_text(inputs:dict):
    return "{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in inputs.items()) + "}"
