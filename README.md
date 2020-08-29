# Answer-Aware Question Generation


## Project Details
In answer-aware question generation, a model is given a passage and a set of candidate answers and tasked with automatically generating questions for each answer. Models like UniLM and ProphetNet provide some benchmarks for these tasks, but we will specifically focus on pre-trained models like T5, GPT-2 and BART.

### answer aware question generation

This package provides scripts that simplify data pipelines, model training and eval with SQuAD-like training data.

## Tasks
We define two types of tasks for this model: question generation (qg) and answer extraction (ans_ext). Tasks are given as inputs to certain parts of the pipeline to specify the format of inputs and outputs to the model.

## Input formats
There are two modes of input formatting: highlight and prefix. highlight is the default. for T5 models, an additional parameter is used to specify the task (e.g. "extract answers" in the case of answer extraction; "generate questions" in the case of question generation)

## Training

To train a question generation model, the answer extraction model must be trained independently of the question generation model. This is the procedure for training either model:

1. Generate Dataset
run `python generate_squad.py`
- the raw SQuAD input files for train/dev are located in ./support
Outputs:
- task data: train_task_data.pt contains the input and target data for both ans_ext and qg tasks
- references: e.g. train_ans_ext_references contains the pre-formatted inputs for the training set with the ans_ext task. train_qg_references contains the pre-formatted inputs for the training set with the question generation task.

2. Prepare Model Inputs
run `python prepare_data.py`
- the data_args.json file given in this directory contains all of the relevant inputs
Outputs:
- featurized training data it pt format for the task specified in the data_args.json input file

3. Training
run `python train.py`
- the train_ag_args.json file given in this directory contains all of the relevant hyperparams for the ans_extraction task. Similarly, the question generation model may be trained by changing the training dataset target in the training_args file
Outputs:
- Model checkpoints in the ./outputs dir

4. Eval
run `python eval.py`
- the eval_ag_args.json file contains all of the relevant eval parameters for the ans_ext task. Similarly, the question generation (qg) tasks may be replace here.
Outputs:
- results: question generated for each dev example. May be compared with the dev_qg_references.txt from step 1.
