import itertools
import os
import random
from typing import List, Tuple

import pandas as pd
from datasets import load_from_disk, Dataset
from datasets import ClassLabel

class TEXTSiameseDataSetConstructor():

    """

    TODO: We want to change this to a script that creates a dataset for TEXT classification, rather than anything else:

    tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("jkhan447/sarcasm-detection-Bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("jkhan447/sarcasm-detection-Bert-base-uncased")

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-topic-sentiment-latest")
    model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-topic-sentiment-latest")

    TOPICS: (not for sentiment)

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/tweet-topic-21-multi")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/tweet-topic-21-multi")

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("dstefa/roberta-base_topic_classification_nyt_news")
    model = AutoModelForSequenceClassification.from_pretrained("dstefa/roberta-base_topic_classification_nyt_news")

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("andreas122001/roberta-academic-detector")
    model = AutoModelForSequenceClassification.from_pretrained("andreas122001/roberta-academic-detector")

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("howanching-clara/classifier_for_academic_texts")
    model = AutoModelForSequenceClassification.from_pretrained("howanching-clara/classifier_for_academic_texts")

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained("allenai/vila-scibert-cased-s2vl")
    model = AutoModelForTokenClassification.from_pretrained("allenai/vila-scibert-cased-s2vl")

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained("allenai/vila-roberta-large-s2vl-internal")
    model = AutoModelForTokenClassification.from_pretrained("allenai/vila-roberta-large-s2vl-internal")


    # Load model directly
    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained("allenai/hvila-row-layoutlm-finetuned-docbank")

    The above is an example of a model that we could use.

    we will also want to combine pos/neg and sarcasm into here for a wider range of emotions.

    So, we want to construct 5-10 different datasets based off the following categories: Each one will be used for lasso
    selection:

    1 Sentiment Analysis: Determining the sentiment or emotional tone of a piece of text, typically classifying it as positive, negative, or neutral. This is commonly used in analyzing customer reviews, social media posts, and survey responses.

    2 Spam Detection: Identifying and filtering out unwanted or unsolicited messages, such as spam emails or spam comments on social media platforms.

    3 Topic Categorization: Assigning predefined categories or labels to text documents based on their content. This is useful in organizing and classifying news articles, research papers, or web content into topics like politics, sports, science, etc.

    4 Language Identification: Determining the language in which a text is written. This is essential for multilingual applications and websites, as well as for preprocessing steps in NLP pipelines.

    5 Intent Detection: Identifying the intent behind a user's input, often used in chatbots and virtual assistants to understand user queries and provide appropriate responses.

    6 Abusive Language Detection: Identifying and filtering out text that contains hate speech, offensive language, or abusive content, commonly used in moderating online platforms and social media.

    7 Authorship Attribution: Determining the author of a given text based on writing style and linguistic features. This is used in literary analysis, forensic linguistics, and plagiarism detection.

    8 Fake News Detection: Identifying and flagging news articles or posts that contain false or misleading information, an increasingly important task in the era of social media and online news.

    9 Emotion Detection: Classifying text based on the emotions expressed, such as joy, anger, sadness, or surprise. This is used in customer feedback analysis, social media monitoring, and psychological research.

    10 Legal Document Classification: Categorizing legal documents into specific types or areas of law, such as contracts, patents, or court opinions, to assist in legal research and case management.


    """

    def __init__(self, batch_size=32, max_appearances=2, max_dataset_size=1000000):
        self.features = None
        self.id2label = None
        self.label2id = None
        self.batch_size = batch_size
        self.max_appearances = max_appearances
        self.max_dataset_size = max_dataset_size

    def run_siamese_dataset_construction_pipeline(self, directory: str, save_directory: str):
        # Load dataset from disk
        dataset_dict = self.load_from_disk(directory)

        # Process each split in the dataset
        for split_name, dataset in dataset_dict.items():
            # Convert the dataset to a Pandas DataFrame
            df = pd.DataFrame(dataset)

            # Partition the DataFrame by label
            partitions = {label: df[df['label'] == label] for label in df['label'].unique()}

            # Calculate the maximum number of pairs per label
            max_pairs_per_label = self.max_dataset_size // (2 * len(partitions))

            print(f"Number of labels: {len(partitions)}")
            print(f"Maximum pairs per label: {max_pairs_per_label}")

            all_sentence_pairs = []

            for label, partition in partitions.items():
                num_pairs = min(len(partition) * (len(partition) - 1) // 2, max_pairs_per_label)
                print(f"Label '{label}': {len(partition)} examples, aiming for {num_pairs} similar pairs")

                pair_count = 0

                for batch in self.generate_sentence_pairs(partition):
                    for row1, row2 in batch:
                        if pair_count >= num_pairs:
                            break
                        pair = {
                            'sentence1': row1['text'],
                            'sentence2': row2['text'],
                            'label': 'similar'
                        }
                        all_sentence_pairs.append(pair)

                        # Generate a dissimilar pair
                        dissimilar_label = random.choice([l for l in partitions.keys() if l != label])
                        dissimilar_row = partitions[dissimilar_label].sample(1).iloc[0]
                        dissimilar_pair = {
                            'sentence1': row1['text'],
                            'sentence2': dissimilar_row['text'],
                            'label': 'dissimilar'
                        }
                        all_sentence_pairs.append(dissimilar_pair)

                        pair_count += 1

                    if pair_count >= num_pairs:
                        break

                print(f"Label '{label}': Generated {pair_count} similar pairs and {pair_count} dissimilar pairs")

            # Save the processed dataset to disk for the current split
            split_save_directory = os.path.join(save_directory, split_name)
            self.save_to_disk(all_sentence_pairs, split_save_directory)

            print(f"Final dataset size: {len(all_sentence_pairs)} pairs")

    def generate_sentence_pairs(self, partition):
        """
        Generate batches of sentence pairs from a partition of the dataset.
        """
        batch = []
        sentences = partition['text'].tolist()

        for i, sentence1 in enumerate(sentences):
            for j, sentence2 in enumerate(sentences):
                if i != j:
                    batch.append(({'text': sentence1}, {'text': sentence2}))

                    # Yield the batch if it reaches the specified size
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []

        # Yield any remaining pairs in the last batch
        if batch:
            yield batch

    def apply_lower_case_augmentation(self, tokens: List[str]) -> List[str]:
        """
        Randomly lowercases tokens in a sentence.
        :param tokens: List of tokens in a sentence.
        :return: List of tokens with random lowercasing applied.
        """
        return [token.lower() if random.choice([True, False]) else token for token in tokens]


    def load_from_disk(self, directory: str):
        """
        Load the dataset from a specified directory on disk.
        :param directory: The directory where the dataset is stored.
        :return: Loaded dataset.
        """
        try:
            dataset = load_from_disk(directory)
            return dataset
        except Exception as e:
            raise IOError(f"Error loading dataset from {directory}: {e}")


    def load_dataset(self, dataset):
        """
        Load and set the dataset features.
        """
        self.features = dataset.features
        self.construct_label_mappings()

    def save_to_disk(self, processed_data, save_path):
        """
        Save the processed data to disk.
        :param processed_data: List of dictionaries, each representing a data row.
        :param save_path: Path where the data will be saved.
        """
        # Convert list of dictionaries to a dictionary of lists
        dataset_dict = {
            'sentence1': [item['sentence1'] for item in processed_data],
            'sentence2': [item['sentence2'] for item in processed_data],
            'label': [item['label'] for item in processed_data]
        }

        # Create a dataset from the dictionary and save it to disk
        processed_dataset = Dataset.from_dict(dataset_dict)
        processed_dataset.save_to_disk(save_path)  # Save the processed dataset to disk

    def construct_label_mappings(self):
        """
        Construct label mappings based on dataset features.
        """
        label_feature = self.features['label']
        if isinstance(label_feature, ClassLabel):
            num_classes = len(label_feature.names)
            self.id2label = {i: label_feature.int2str(i) for i in range(num_classes)}
            self.label2id = {v: k for k, v in self.id2label.items()}
        else:
            raise ValueError("Label feature is not a ClassLabel type.")



# Create an instance of the class
dataset_constructor = TEXTSiameseDataSetConstructor()

# Here is the path to the dataset that we want to load from disk.
dataset_path = r"C:\Users\doren\AppData\Roaming\Gantrithor\data\datasets\saved_dataset_sentiment_copy"

# Define a path where you want to save the processed Siamese dataset
save_directory = r"C:\Users\doren\AppData\Roaming\Gantrithor\data\datasets\sbert_dataset_saimese_encoder"

# Run the Siamese dataset construction pipeline
dataset_constructor.run_siamese_dataset_construction_pipeline(directory=dataset_path, save_directory=save_directory)



from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader, Dataset
import os
from datasets import load_dataset
import torch.nn as nn

class InputExampleDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example.texts[0], example.texts[1], example.label

class SBERTTrainer:
    """
    A trainer class for fine-tuning a Sentence-BERT (SBERT) model for sentence similarity tasks.

    Attributes:
        dataset_directory (str): The directory where the training and validation datasets are stored.
        model_name (str): The name or path of the pre-trained transformer model to be used as the base for the SBERT model.
        batch_size (int): The batch size for training and evaluation.
        epochs (int): The number of epochs to train the model.
        evaluation_steps (int): The number of training steps after which the model is evaluated on the validation set.
        output_path (str): The path where the fine-tuned model will be saved.
        max_seq_length (int): The maximum sequence length for the input text. Longer sequences will be truncated.
        dense_out_features (int): The number of output features for the dense layer.

    The SBERT model consists of three main components:
        1. Word Embedding Model (models.Transformer): This is the base transformer model (e.g., BERT, RoBERTa)
        that generates contextualized word embeddings for the input text.
        The `max_seq_length` attribute controls the maximum length of the input sequences.

        2. Pooling Model (models.Pooling): This layer pools the word embeddings generated by the transformer model to create a fixed-size sentence embedding.
        The pooling strategy (e.g., mean pooling) is determined by the configuration of the Pooling model.

        3. Dense Model (models.Dense): This is a fully connected dense layer that projects the pooled sentence embeddings to a lower-dimensional space.
        The `dense_out_features` attribute specifies the number of output features for this layer.
        The activation function (e.g., Tanh) can be specified in the configuration of the Dense model.

    The `train` method fine-tunes the SBERT model on the training dataset using the specified loss function (e.g., CosineSimilarityLoss)
    and evaluates the model on the validation dataset using an evaluator (e.g., EmbeddingSimilarityEvaluator).

    """
    def __init__(self, dataset_directory, model_name='microsoft/mpnet-base', batch_size=32, epochs=2, evaluation_steps=1000,
                 output_path='output/sbert_model', max_seq_length=128, dense_out_features=256):
        self.dataset_directory = dataset_directory
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.evaluation_steps = evaluation_steps
        self.output_path = output_path
        self.max_seq_length = max_seq_length
        self.dense_out_features = dense_out_features

        # Customizing the model components
        self.word_embedding_model = models.Transformer(self.model_name, max_seq_length=self.max_seq_length)
        self.pooling_model = models.Pooling(self.word_embedding_model.get_word_embedding_dimension())
        self.dense_model = models.Dense(
            in_features=self.pooling_model.get_sentence_embedding_dimension(),
            out_features=self.dense_out_features,
            activation_function=nn.Tanh()
        )

    def load_dataset(self, split):
        dataset = load_dataset('arrow',
                               data_files=os.path.join(self.dataset_directory, split, 'data-00000-of-00001.arrow'))
        examples = []
        for item in dataset[split]:
            examples.append(
                InputExample(texts=[item['sentence1'], item['sentence2']], label=float(item['label'] == 'similar')))
        return examples

    def train(self):
        train_examples = self.load_dataset('train')
        dev_examples = self.load_dataset('validation')

        train_dataset = InputExampleDataset(train_examples)

        # Constructing the SentenceTransformer model with custom components
        model = SentenceTransformer(modules=[self.word_embedding_model, self.pooling_model, self.dense_model])

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        train_loss = losses.CosineSimilarityLoss(model)

        # Extract sentences and scores for the evaluation dataset
        sentences1 = [example.texts[0] for example in dev_examples]
        sentences2 = [example.texts[1] for example in dev_examples]
        scores = [example.label for example in dev_examples]

        evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluator,
                  epochs=self.epochs,
                  evaluation_steps=self.evaluation_steps,
                  output_path=self.output_path)

        model.save(self.output_path)

# Example usage
dataset_path = r"C:\Users\doren\AppData\Roaming\Gantrithor\data\datasets\sbert_dataset_saimese_encoder"
trainer = SBERTTrainer(dataset_directory=dataset_path,
                       batch_size=32,
                       epochs=3,
                       evaluation_steps=1000,
                       output_path='output/sbert_model',
                       max_seq_length=256,
                       dense_out_features=256)
trainer.train()
