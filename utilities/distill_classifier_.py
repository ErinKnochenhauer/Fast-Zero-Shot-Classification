import torch
from datasets import Dataset
from tqdm.auto import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer
)


DESCRIPTION = """
Distills an NLI-based zero-shot classifier to a smaller, more efficient model with a fixed set of candidate class
names. Useful for speeding up zero-shot classification in cases where labeled training data is not available, but
when only a single fixed set of classes is needed. Takes a teacher NLI model, student classifier model, unlabeled
dataset, and set of K possible class names. Yields a single classifier with K outputs corresponding to the provided
class names.
"""


class ZeroShotStudentTrainer:
    def __init__(self, examples, class_names, hypothesis_template="This text is about {}."):
        self.examples=examples
        self.class_names=class_names
        self.hypothesis_template=hypothesis_template


    def get_premise_hypothesis_pairs(self):
        premises = []
        hypotheses = []
        for example in self.examples:
            for name in self.class_names:
                premises.append(example)
                hypotheses.append(self.hypothesis_template.format(name))
        return premises, hypotheses

    def get_teacher_predictions(self,
                                model_path="roberta-large-mnli",
                                batch_size=32,
                                temperature=1.0,
                                multi_label=False,
                                use_fast_tokenizer=False,
                                fp16=False
                                ):
        """
        Gets predictions by the same method as the zero-shot pipeline but with DataParallel
        & more efficient batching
        """
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model_config = model.config

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast_tokenizer)

        premises, hypotheses = self.get_premise_hypothesis_pairs()
        logits = []

        for i in tqdm(range(0, len(premises), batch_size)):
            batch_premises = premises[i : i + batch_size]
            batch_hypotheses = hypotheses[i : i + batch_size]

            encodings = tokenizer(
                    batch_premises,
                    batch_hypotheses,
                    padding=True,
                    truncation="only_first",
                    return_tensors="pt",
                )

            with torch.cuda.amp.autocast(enabled=fp16):
                with torch.no_grad():
                    outputs = model(**encodings)
            logits.append(outputs.logits.detach().cpu().float())

        entail_id = get_entailment_id(model_config)
        contr_id = -1 if entail_id == 0 else 0
        logits = torch.cat(logits, dim=0)  # N*K x 3
        nli_logits = logits.reshape(len(self.examples),
                                    len(self.class_names), -1)[..., [contr_id, entail_id]]  # N x K x 2

        if multi_label:
            # softmax over (contr, entail) logits for each class independently
            nli_prob = (nli_logits / temperature).softmax(-1)
        else:
            # softmax over entail logits across classes s.t. class probabilities sum to 1.
            nli_prob = (nli_logits / temperature).softmax(1)

        return nli_prob[..., 1]  # N x K

    def distill_text_classifier(self, training_args):
        """
        Main function for training a smaller student model based on labels from a pretrained
        teacher model.
        """
        # get teacher predictions and load into a dataset
        print("Generating predictions from zero-shot teacher model")
        teacher_soft_preds = self.get_teacher_predictions()

        dataset = Dataset.from_dict(
            {
                "text": self.examples,
                "labels": teacher_soft_preds,
            }
        )

        # create student model
        print("Initializing student model")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(self.class_names)
        )

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",
                                                use_fast=False)
        
        model.config.id2label = dict(enumerate(self.class_names))
        model.config.label2id = {label: i for i, label in enumerate(self.class_names)}

        print("Splitting dataset into training and testing")
        ds = dataset.train_test_split(test_size=0.1, seed=48)
        print(ds)

        print("Tokenizing training and testing datasets")
        ds = ds.map(tokenizer, input_columns="text")
        ds.set_format("torch")
        print(ds)

        # define training setup
        trainer = Trainer(model=model,
                          tokenizer=tokenizer,
                          args=training_args,
                          train_dataset=ds['train'],
                          eval_dataset=ds['test'],
                          compute_metrics=compute_metrics)

        print("Training student model on teacher predictions")
        trainer.train()

        agreement = trainer.evaluate(eval_dataset=ds['test'])["eval_agreement"]
        print(f"Agreement of student and teacher predictions: {agreement * 100:0.2f}%")

        trainer.save_model()


# class TextClassifer:
#     def __init__(self, pipeline, class_names):

#         self._pipeline = pipeline
#         self.class_names = class_names

#     def classify_text(self,
#                       text,
#                       batch_size=32,
#                       hypothesis_template="This text is about {}."):
#         """
#         Uses a pretrained classifier to return the predicted class labels
#         for each example
#         """
#         start = time()
#         preds = []
#         for i in tqdm(range(0, len(text), batch_size)):
#             abatch = text[i:i+batch_size]
#             outputs = self._pipeline(abatch, self.class_names, hypothesis_template)
#             try:
#                 preds += [self.class_names.index(o['label'][0]) for o in outputs]
#             except:
#                 preds += [self.class_names.index(o['labels'][0]) for o in outputs]


#         print(f"Runtime: {time() - start : 0.2f} seconds")
        
#         return preds


def read_lines(path):
    """Read in text file"""
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                lines.append(line)
    return lines


def compute_metrics(p, return_outputs=False):
    preds = p.predictions.argmax(-1)
    proxy_labels = p.label_ids.argmax(-1)  # "label_ids" are actually distributions
    return {"agreement": (preds == proxy_labels).mean().item()}


def get_entailment_id(config):
    for label, ind in config.label2id.items():
        if label.lower().startswith("entail"):
            return ind
    print("Could not identify entailment dimension from teacher config label2id. Setting to -1.")
    return -1


def get_results_df(text, class_names, preds, results_path=None):
    """Get a pandas df of text labeled with class names"""
    import pandas as pd
    mapping = {k: v for k, v in enumerate(class_names)}
    results = pd.DataFrame(
        {
            'text': text,
            'label': preds
        }
    )

    results['label'] = results['label'].map(mapping)

    if results_path:
        results.to_csv(results_path, index=False)
    return results
