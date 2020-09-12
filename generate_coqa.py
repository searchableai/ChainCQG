import os
import json

class CoqaFeaturizer:
    def __init__(self):
        self._train_path, self._dev_path = self._split_generator()

    def __call__(self):
        return _

    def _split_generator(self):
        split_labels = {
            "train": os.path.join("support", "coqa-train-v1.0.json"),
            "dev": os.path.join("support", "coqa-dev-v1.0.json")
        }
        return [
            split_labels['train'],
            split_labels['dev']
        ]

    def _generate_examples(self, datapath, split):
        # Yields (key, example) tuples from the dataset
        print('Processing file: ', datapath)
        with open(datapath, encoding="utf-8") as f:
            data = json.load(f)
            for row in data["data"]:
                questions = [question["input_text"] for question in row["questions"]]
                story = row["story"]
                source = row["source"]
                answers_start = [answer["span_start"] for answer in row["answers"]]
                answers_end = [answer["span_end"] for answer in row["answers"]]
                answers = [answer["input_text"] for answer in row["answers"]]
                yield row["id"], {
                    "source": source,
                    "story": story,
                    "questions": questions,
                    "answers": {"input_text": answers, "answer_start": answers_start, "answer_end": answers_end},
                }


if __name__ == "__main__":
    featurizer = CoqaFeaturizer()
    train_path, dev_path = featurizer._split_generator()
    featurizer._generate_examples(train_path, 'train')
    featurizer._generate_examples(dev_path, 'dev')
