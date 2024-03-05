import os
import random

import torch
import torch.nn as nn
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader, Dataset



class InputExampleDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SBERTTrainer:
    """
    We may eventually want the dataset size to be streamed (load up a parametrized slice of it).
    potential model_names and their training speeds and model size:


    1 microsoft/mpnet-base    ~5 it/s
    2 prajjwal1/bert-tiny     ~ 62 it/s
    3 google/mobilebert-uncased 4 it/s

    """

    def __init__(self, dataset_directory, model_name='prajjwal1/bert-tiny', batch_size=32, epochs=2, evaluation_steps=1000,
                 output_path='output/sbert_model', max_seq_length=128, dense_out_features=256):
        self.dataset_directory = dataset_directory
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.evaluation_steps = evaluation_steps
        self.output_path = output_path
        self.max_seq_length = max_seq_length
        self.dense_out_features = dense_out_features

        self.word_embedding_model = models.Transformer(self.model_name, max_seq_length=self.max_seq_length)
        self.pooling_model = models.Pooling(self.word_embedding_model.get_word_embedding_dimension())
        self.dense_model = models.Dense(
            in_features=self.pooling_model.get_sentence_embedding_dimension(),
            out_features=self.dense_out_features,
            activation_function=nn.Tanh()
        )

    def load_dataset(self, split):
        dataset_path = os.path.join(self.dataset_directory, split)
        dataset = load_from_disk(dataset_path)
        examples = []
        for item in dataset:
            examples.append(
                InputExample(texts=[item['sentence1'], item['sentence2']], label=float(item['label'] == 'similar')))
        return examples

    def get_random_evaluation_examples(self, examples, num_examples=20):
        return random.sample(examples, num_examples)

    def train(self):
        train_examples = self.load_dataset('train')

        # Get random evaluation examples from the training dataset
        eval_examples = self.get_random_evaluation_examples(train_examples)

        # Create evaluation sentence lists
        sentences1 = [example.texts[0] for example in eval_examples]
        sentences2 = [example.texts[1] for example in eval_examples]
        scores = [example.label for example in eval_examples]

        train_dataset = InputExampleDataset(train_examples)
        model = SentenceTransformer(modules=[self.word_embedding_model, self.pooling_model, self.dense_model])

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        train_loss = losses.CosineSimilarityLoss(model)

        evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluator,
                  epochs=self.epochs,
                  evaluation_steps=self.evaluation_steps,
                  output_path=self.output_path)

        model.save(self.output_path)

# Example usage
dataset_path = r"C:\Users\doren\AppData\Roaming\Gantrithor\data\datasets\sbert_dataset_saimese_encoder"

output_path = r"C:\Users\doren\AppData\Roaming\Gantrithor\data\sentence_encoders\sentiment_encoder_128_256"
output_path_512 = r"C:\Users\doren\AppData\Roaming\Gantrithor\data\sentence_encoders\sentiment_encoder_512"
output_path_64 = r"C:\Users\doren\AppData\Roaming\Gantrithor\data\sentence_encoders\sentiment_encoder_64"
output_path_32 = r"C:\Users\doren\AppData\Roaming\Gantrithor\data\sentence_encoders\sentiment_encoder_32"


trainer = SBERTTrainer(dataset_directory=dataset_path,
                       batch_size=128,
                       epochs=1,
                       evaluation_steps=2000,
                       output_path=output_path,
                       max_seq_length=128,
                       dense_out_features=256)
trainer.train()