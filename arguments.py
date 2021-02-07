import argparse


def add_generate_args(args_manager):

    args_manager.add_argument("--random_seed",
                              type=int,
                              help="random_seed")

    args_manager.add_argument("--dropout_times",
                              default="",
                              type=int,
                              help="dropout times for variance calculation")

    args_manager.add_argument("--log_comment",
                              default="",
                              help="The logging directory")

    args_manager.add_argument(
        "--max_gen_seq_length",
        default=48,
        type=int,
        help="The maximum sequence length for generation")

    args_manager.add_argument("--train_temperature",
                              default=0.01,
                              type=float,
                              help="Temperature used in generate training samples")
    
    args_manager.add_argument("--test_temperature",
                              default=0.01,
                              type=float,
                              help="Temperature used in generate evaluation sample")
    

    args_manager.add_argument("--temperature",
                              default=1.0,
                              type=float,
                              help="Temperature setting for softmax")
    args_manager.add_argument("--top_p",
                              default=0.9,
                              type=float,
                              help="Top p for nucleus sampling")
    args_manager.add_argument("--top_k",
                              default=-1,
                              type=float,
                              help="Top k for nucleus sampling")
    args_manager.add_argument("--sam",
                              default=-1,
                              type=float,
                              help="Top k for nucleus sampling")
    args_manager.add_argument("--model_path",
                            default="",
                            type=str,
                            help="Model Path")
    # base model or not
    args_manager.add_argument("--use_base",
                                action='store_true',
                                help="Use base model or not")

    args_manager.add_argument("--use_drop",
                                action="store_true",
                                help="use dropout on mm or not")

    args_manager.add_argument("--use_train",
                                action="store_true",
                                help="use training set for training or not")

    args_manager.add_argument("--dataset_name",
                                default="",
                                type=str,
                                help="persuasion, dailydialog, personachat, dstc-7, dstc-7-sampled, rocstory")

    args_manager.add_argument("--task_type",
                                default="",
                                type=str,
                                help="dialog or story")

    args_manager.add_argument("--gp_lambda",
                                default="1.0",
                                type=float,
                                help="lambda for gradient penalty")
    args_manager.add_argument("--uncertain_lambda",
                                default="1.0",
                                type=float,
                                help="initial lambda for uncertain loss")
    args_manager.add_argument("--uncertain_beta",
                            default="0.5",
                            type=float,
                            help="target for uncertain loss")

    args_manager.add_argument("--uncertain_factor",
                                default="0.9",
                                type=float,
                                help="used to adjust lambda in uncertain loss")

            


def add_ppo_args(args_manager):
    args_manager.add_argument("--model_size",
                      default="small",
                      type=str,
                      help="Model size")
    args_manager.add_argument("--buffer_size",
                              default=1024,
                              type=int,
                              help="PPO Rollout Buffer Size")

    args_manager.add_argument("--eval_batch_size",
                              default=16,
                              type=int,
                              help="Evaluation batch size")

    args_manager.add_argument("--mini_batch_size",
                              default=16,
                              type=int,
                              help="RL mini batch size")

    args_manager.add_argument("--generator_model_path",
                              default="",
                              type=str,
                              help="The default generator's path")

    args_manager.add_argument("--language_model_path",
                              default="",
                              type=str,
                              help="The language model's weights path")
