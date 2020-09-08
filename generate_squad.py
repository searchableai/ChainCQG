# Lint as: python3
"""SQUAD: The Stanford Question Answering Dataset."""

from __future__ import absolute_import, division, print_function

import json
import jsonlines
import logging
import os

import nltk
nltk.download('punkt')

QG_FORMATS = [
    "prepend",
    "highlight",
    "prepend_highlight",
]


class SquadFeaturizer:
    def __init__(self, qg_format="highlight", data_format="squadv1", ):
        self.qg_format = qg_format
        self._train_path, self._dev_path = self._split_generator(data_format)
        self._data_format = data_format

    def __call__(self):
        return _

    def _split_generator(self, data_format):
        self._data_format = data_format
        if data_format == "squadv1":
            split_labels = {
                "train": os.path.join("support", "train-v1.1.json"),
                "dev": os.path.join("support", "dev-v1.1.json")
            }
        elif data_format == "squadv2":
            split_labels = {
                "train": os.path.join("support", "train-v2.0.json"),
                "dev": os.path.join("support", "dev-v2.0.json")
            }
        return [
            split_labels['train'],
            split_labels['dev']
        ]
    
    def _get_correct_alignment(self, context, answer):
        """ Some original examples in SQuAD have incorrect indices. We test and fix this here. """
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx       # When the gold label position is correct
        elif context[start_idx-1:end_idx-1] == gold_text:
            return start_idx-1, end_idx-1   # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            return start_idx-2, end_idx-2   # When the gold label is off by two character
        else:
            raise ValueError()
    
    def process_qg_text(self, context, question, answer):
        answer_text = answer['text'].strip()
        
        if self.qg_format == "prepend":
            q_gen_input = f"answer: {answer_text}  context: {context}"
        elif self.qg_format == "highlight":
            start_pos, end_pos = self._get_correct_alignment(context, answer)
            q_gen_input = f"generate question: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"
        else:
            start_pos, end_pos = self._get_correct_alignment(context, answer)
            q_gen_input = f"answer: {answer_text} context: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"
        
        q_gen_target = f"{question}"
        return {"source_text": q_gen_input, "target_text": q_gen_target, "task": "qg"}
    
    def process_ans_ext(self, paragraph):
        context = paragraph['context'].strip()
    
        # split into sentences
        sents = nltk.sent_tokenize(context)

        # get positions of the sentences
        positions = []
        for i, sent in enumerate(sents):
            if i == 0:
                start, end = 0, len(sent)
            else:
                start, end = (prev_end + 1), (prev_end + len(sent) + 1)
            prev_end = end
            positions.append({'start': start, 'end': end})
        
        # get answers
        if self._data_format == 'squadv1':
            answers = [qa['answers'][0] for qa in paragraph['qas']]
        elif self._data_format == 'squadv2':
            answers = [qa['answers'][0] if qa['answers'] else {'text':'', 'answer_start':0} for qa in paragraph['qas']]

        # get list of answers for each sentence
        sent_answers = []
        for pos, sent in zip(positions, sents):
            target_answers = []
            for ans in answers:
                if ans['answer_start'] in range(pos['start'], pos['end']):
                    target_answers.append(ans['text'].strip())
            sent_answers.append(target_answers)

        # build inputs and targets
        examples = []
        for i, ans in enumerate(sent_answers):
            context = "extract answers:"
            if len(ans) == 0: continue
            ans = list(set(ans))
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "{hl_token} %s {hl_token}" % sent
                context = "%s %s" % (context, sent)
                context = context.strip()
            input_text = context
            target_text = " {sep_token} ".join(ans) + " {sep_token}"

            examples.append({'source_text': input_text, "target_text": target_text, "task": "ans_ext"})
        
        return examples

    def _generate_examples(self, datapath, prefix):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", datapath)
        count = 0
        tasks = ['qg', 'ans_ext']
        qg_inputs = []
        ans_ext_inputs = []
        references = []
        task_data = {k:[] for k in tasks}
        print('Processing file: ', datapath)
        with open(datapath) as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    
                    if 'ans_ext' in tasks:
                        ans_ext_examples = self.process_ans_ext(paragraph)
                        for example in ans_ext_examples:
                            task_data['ans_ext'].append(example)
                            ans_ext_inputs.append(example)
                    
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        references.append(question)
                        id_ = qa["id"]

                        answers = [answer["text"].strip() for answer in qa["answers"]]
                        for task in tasks:
                            if task == 'qg':
                                if self._data_format == 'squadv2':
                                    ans = qa['answers'][0] if qa['answers'] else {'text':'', 'answer_start':0}
                                    example = self.process_qg_text(context, question, ans)
                                elif self._data_format == 'squadv1':
                                    example = self.process_qg_text(context, question, qa["answers"][0])
                                task_data[task].append(example)
                                qg_inputs.append(example)
        print('Processed {} input examples - {} ans_ext - {} qg'.format(len(qg_inputs), len(task_data['ans_ext']), len(task_data['qg'])))
        with open(prefix+'_task_data.json', 'w') as f:
            json.dump(task_data, f)
        with jsonlines.open(prefix+'_qg_inputs.txt', 'w') as f:
            f.write(qg_inputs)
        with jsonlines.open(prefix+'_ans_ext_inputs.txt', 'w') as f:
            f.write(ans_ext_inputs)
        with open(prefix+'_ans_ext_references.txt', 'w') as f:
            for ref in ans_ext_inputs:
                f.write(ref['target_text']+'\n')
        with open(prefix+'_qg_references.txt', 'w') as f:
            for ref in references:
                f.write(ref+'\n')

if __name__ == "__main__":
    featurizer = SquadFeaturizer()
    train_path, dev_path = featurizer._split_generator(data_format='squadv1')
    featurizer._generate_examples(train_path, 'train')
    featurizer._generate_examples(dev_path, 'dev')
