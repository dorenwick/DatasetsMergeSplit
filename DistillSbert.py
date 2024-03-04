import logging
import os
import gzip
import csv
import random
import numpy as np
import torch
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, models, util
from sentence_transformers import evaluation, InputExample

class EmbeddingDimensionReducer:
    def __init__(self, parent_model_name, student_model=None, target_dim=128, dataset_path=None):
        self.parent_model = SentenceTransformer(parent_model_name)
        self.student_model = student_model if student_model else self.parent_model
        self.target_dim = target_dim
        self.dataset_path = dataset_path or "datasets/AllNLI.tsv.gz"
        self._prepare_dataset()

    def _prepare_dataset(self):
        if not os.path.exists(self.dataset_path):
            util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", self.dataset_path)

    def reduce_dimension_and_evaluate(self):
        self.reduce_dimension()
        score = self.evaluate_model()
        return score

    def evaluate_model(self):
        # Ensure the STS benchmark dataset is available
        sts_dataset_path = "datasets/stsbenchmark.tsv.gz"
        if not os.path.exists(sts_dataset_path):
            util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)

        eval_examples = []
        with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
            reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if row["split"] == "test":
                    score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
                    eval_examples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=score))

        # Evaluate the model
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(eval_examples, name="sts-benchmark-test")
        return evaluator(self.student_model)
    def reduce_dimension(self):
        sentences = self._read_sentences_from_dataset()
        train_embeddings = self.parent_model.encode(sentences, convert_to_numpy=True)

        # Compute PCA
        pca = PCA(n_components=self.target_dim)
        pca.fit(train_embeddings)
        pca_comp = np.asarray(pca.components_)

        # Add dense layer for dimensionality reduction
        dense = models.Dense(
            in_features=self.parent_model.get_sentence_embedding_dimension(),
            out_features=self.target_dim,
            bias=False,
            activation_function=torch.nn.Identity(),
        )
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
        self.student_model.add_module("dense", dense)

    def _read_sentences_from_dataset(self):
        sentences = set()
        with gzip.open(self.dataset_path, "rt", encoding="utf8") as fIn:
            reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                try:
                    sentence1 = row["sentence1"]
                    sentence2 = row["sentence2"]
                    sentences.add(sentence1)
                    sentences.add(sentence2)
                except KeyError as e:
                    logging.error(f"Key error: {e}. Row: {row}")
                    continue

        sentences = list(sentences)
        random.shuffle(sentences)
        return sentences[:100000]  # Select a subset for PCA training

    def write_readme(self):
        """
        The user can do something like this:

        # Directory for the 16-dimensional model
        base_model_directory = r"C:\Users\doren\AppData\Roaming\Gantrithor\data\sentence_encoders"
        model_dim = 32
        model_directory = os.path.join(base_model_directory, f'reduced_model_{model_dim}', f'model-{model_dim}dim')
        or model_directory = "ClovenDoug/small_128_all-MiniLM-L6-v2"
        model = SentenceTransformer(model_directory)

        :return:
        """
        pass

    def save_model(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.student_model.save(os.path.join(directory, f'model-{self.target_dim}dim'))


target_dimensions = [256, 128, 64, 48, 32, 16]

for dim in target_dimensions:
    reducer = EmbeddingDimensionReducer("all-MiniLM-L6-v2", target_dim=dim)
    score = reducer.reduce_dimension_and_evaluate()
    print(f"{dim} Dim Model STS Score:", score)
    reducer.save_model(f'models/reduced_model_{dim}')



