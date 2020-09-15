import os
import json

class QuacFeaturizer:
    def __init__(self):
        self._train_path, self._dev_path = self._split_generator()

    def __call__(self):
        return _

    def _split_generator(self):
        split_labels = {
            "train": os.path.join("support", "quac-train-v0.2.json"),
            "dev": os.path.join("support", "quac-dev-v0.2.json")
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
        for article in data["data"]:
            title = article.get("title", "").strip()
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    yield qa["id"], {
                        "question": qa["question"].strip(),
                        "followup": True if qa["followup"] == 'y' else False,
                        "yesno": True if qa["yesno"] == 'y' else False,
                        "answers":[answer["text"].strip() for answer in qa["answers"]]
                    }


if __name__ == "__main__":
    featurizer = QuacFeaturizer()
    train_path, dev_path = featurizer._split_generator()
    featurizer._generate_examples(train_path, 'train')
    featurizer._generate_examples(dev_path, 'dev')
