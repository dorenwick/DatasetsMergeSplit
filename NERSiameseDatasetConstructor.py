import itertools
import json
import os
import random
from typing import List, Tuple
from datasets import load_from_disk, Dataset
from datasets import ClassLabel
from span_marker import SpanMarkerModel
from transformers import AutoConfig


class NERSiameseDataSetConstructor():

    """
    We will find NER with wikipedia dataset or with textbooks
    from datasets import load_dataset

    dataset = load_dataset("SciPhi/textbooks-are-all-you-need-lite")


    This class DatasetConstructor should be used here to make the siamese dataset from the older dataset.
    The goal here is to make a dataset that reads through the old dataset, and it will read through 2 rows at a time:
    It will extract the tokens from each row, join them into a string, then tokenize them into sentences.
    It shall create corresponding ner_tag sentence lists of integer encodings for the class labels when it does this.

    Then, it will do the following:
    It will compare all pairs of sentences and their ner_tags. If two sentences have no non-zero ner_tags in common,
    then it will categorize them as "dissimilar" sentences in preparation for a dataset to be trained on a siamese
    sentence encoder.

    If two sentences do have non zero integer ner_tags in common, then they will be categorized as similar.
    We also will do the following:
    we will convert the ner_tags into their actual class labels,

    using:

    self.id2label,
    self.id2label_iob,
    self.label2id,
    self.label2id_iob

    Here is an example of how such attributes are structured.

    'id2label': {0: 'O', 1: 'art', 2: 'building', 3: 'event', 4: 'location', 5: 'organization', 6: 'other', 7: 'person', 8: 'product'},
    'label2id': {'O': 0, 'art': 1, 'building': 2, 'event': 3, 'location': 4, 'organization': 5, 'other': 6, 'person': 7, 'product': 8},
    'id2label_iob': {0: 'O', 1: 'B-art', 2: 'I-art', 3: 'B-building', 4: 'I-building', 5: 'B-event', 6: 'I-event', 7: 'B-location', 8: 'I-location', 9: 'B-organization', 10: 'I-organization', 11: 'B-other', 12: 'I-other', 13: 'B-person', 14: 'I-person', 15: 'B-product', 16: 'I-product'},
    'label2id_iob': {'O': 0, 'B-art': 1, 'I-art': 2, 'B-building': 3, 'I-building': 4, 'B-event': 5, 'I-event': 6, 'B-location': 7, 'I-location': 8, 'B-organization': 9, 'I-organization': 10, 'B-other': 11, 'I-other': 12, 'B-person': 13, 'I-person': 14, 'B-product': 15, 'I-product': 16},
    'format_iob_list': ['O', 'B-art', 'I-art', 'B-building', 'I-building', 'B-event', 'I-event', 'B-location', 'I-location', 'B-organization', 'I-organization', 'B-other', 'I-other', 'B-person', 'I-person', 'B-product', 'I-product'],
    'id2reduced_id': {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: 6, 12: 6, 13: 7, 14: 7, 15: 8, 16: 8}

     Now, in addition to labelling two sentences with common ner tags as similar, we shall also create sentences using the id2label values
     of the ner tags, and label them as similar to the sentence itself.

     For example, consider the following 3 sentences. The first is one we find in the dataset.
     The ner tags would be [0, 7, 0, 1, 2] if ner_tags is iob format and [0,7,0,1,1] if they are not iob format.
     If iob format then we would simply convert them to iob format with id2reduced_id. Then we would use id2label on them,
     and we would create sentence 2 and sentence 3 and make them "similar".

     sentence 1: "Chomsky wrote syntactic structures"
     sentence 2: "person"
     sentence 3: "art"

     For each similar sentence we make from a detected entity, we would also want to include a dissimilar entity.
     So, we could make 2 entities that don't appear in them like:

     sentence 4: "building"
     sentence 5: "location"

     and sentence 1 would be labelled as dissimilar to sentence 4 and sentence 5.

     Of course, we may not have that many entities to choose from. If that happens, then skip over this sentence to the next one.

     Here are the details on the dataset features:

    Here is the structure and features of our self.loaded_dataset:

        self.features = Features({
            "tokens": Sequence(feature=Value(dtype="string")),
            "ner_tags": Sequence(ClassLabel(names=self.labels)),
            "document_id": Value(dtype='int32'),
        })

    Our self.siamese_dataset will have the following features:

        self.features = Features({
            'sentence1': Value(dtype='string'),
            'sentence2': Value(dtype='string'),
            'label': ClassLabel(num_classes=2, names=['not_similar', 'similar'])
        })

    So, here is how our class will be structured. Firstly, we will want a method that is a full pipeline that
    uses all the other methods in it.

    we will have a method for loaded dataset, which loads a specified dataset in, and extracts/defines its features.

    we will have a method for constructing id2label, id2label_iob, label2id, label2id_iob, id2reduced_id from these
    features. This method shall include a method for detecting if class labels are iob_format or not. (use all(B-,I-,O) beginning).

    We will have a method for converting lists to strings, then tokenizing stringified tokens to sentences, and then
    making them lists of sentences. This method will take a single row of the dataset as the argument, and return the lists
    of tokens and list of ner tags for each sentence.

    We will have another method that calls upon this for each document/row.

    We will have another method for creating all of the similar/dissimilar sentence pairs for each returned list of tokens / tags for
    each sentence. We may or may not have a method for each particular case of making entity sentences, comparing two original sentences,
    ect.

    TODO:
        Chatgpt, read through this comment string and implement all the details.
        Do not add redundant methods (copies of pre existing methods).
        Keep refining existing methods if they need to be.
        Also, simplify the construct other formats
        from id2label, and make it take the ClassLabels as input, not id2label. And make it check for iob format...once that's done you
        can start making the other formats.



    """

    def __init__(self):
        self.features = None
        self.id2label = None
        self.label2id = None
        self.id2label_iob = None
        self.label2id_iob = None
        self.id2reduced_id = None
        self.model_names = [
            "tomaarsen/span-marker-roberta-large-ontonotes5",
            "tomaarsen/span-marker-bert-base-fewnerd-fine-super",
            "tomaarsen/span-marker-roberta-large-fewnerd-fine-super",
            "tomaarsen/span-marker-bert-tiny-fewnerd-coarse-super",
            "tomaarsen/span-marker-bert-base-uncased-bionlp",
            "tomaarsen/span-marker-bert-base-uncased-keyphrase-inspec",
            "tomaarsen/span-marker-bert-base-ncbi-disease"
            "lambdavi/span-marker-luke-legal"
        ]


    def run_siamese_dataset_construction_pipeline(self, directory: str, save_directory: str,
                                                  augment_lower_case: int = 5):


        # Load dataset from disk
        dataset_dict = self.load_from_disk(directory)

        self.print_model_metadata()

        # Process each split in the dataset
        for split_name, dataset in dataset_dict.items():
            # Load and set dataset features for the current split
            self.load_dataset(dataset)

            # Convert labels to IOB format if needed
            self.convert_to_iob_format()

            all_entities = set()
            all_sentence_pairs = []
            sentence_counter = 0

            for row in dataset:
                tokens, ner_tags = self.process_row(row)

                if sentence_counter % augment_lower_case == 0:
                    tokens = self.apply_lower_case_augmentation(tokens)
                sentence_counter += 1

                self.generate_similar_pairs_for_row(tokens, ner_tags, all_sentence_pairs)
                all_entities.update(self.create_entity_based_sentences(tokens, ner_tags))

            for row in dataset:
                tokens, ner_tags = self.process_row(row)
                self.generate_dissimilar_pairs_for_row(tokens, ner_tags, all_sentence_pairs, all_entities)

            # Save the processed dataset to disk for the current split
            split_save_directory = os.path.join(save_directory, split_name)
            self.save_to_disk(all_sentence_pairs, split_save_directory)

    def print_model_metadata(self):
        all_entities = set()
        print("Model Metadata:\n")

        for model_name in self.model_names:
            try:
                config = AutoConfig.from_pretrained(model_name)
                id2label = config.id2label

                print(f"Model Name: {model_name}")
                print(f"id2label: {id2label}\n")

                all_entities.update(id2label.values())

            except Exception as e:
                print(f"Error loading model {model_name}: {e}\n")

        print("Summary:")
        print(f"Total unique entities across all models: {len(all_entities)}")
        print(f"Entities: {all_entities}\n")

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
        processed_dataset.save_to_disk(save_path)

    def construct_label_mappings(self):
        """
        Construct label mappings based on dataset features.
        """
        ner_tags_feature = self.features['ner_tags'].feature
        if isinstance(ner_tags_feature, ClassLabel):
            num_classes = len(ner_tags_feature.names)
            self.id2label = {i: ner_tags_feature.int2str(i) for i in range(num_classes)}
            self.label2id = {v: k for k, v in self.id2label.items()}
        else:
            raise ValueError("NER tags feature is not a ClassLabel type.")

    def check_iob_format(self):
        """
        Check if the labels are in IOB format.
        """
        return all(label.startswith(('B-', 'I-')) or label == 'O' for label in self.id2label.values())

    def convert_to_iob_format(self):
        """
        Convert labels to IOB format if they are not already in that format.
        """
        if not self.check_iob_format():
            self.id2label_iob = {k: (f"B-{v}" if v != 'O' else 'O') for k, v in self.id2label.items()}
            self.label2id_iob = {v: k for k, v in self.id2label_iob.items()}
            self.id2reduced_id = {k: k for k, v in self.id2label_iob.items()}
        else:
            self.id2label_iob = self.id2label
            self.label2id_iob = self.label2id
            self.id2reduced_id = {k: k for k, v in self.id2label_iob.items()}

    def process_row(self, row) -> Tuple[List[str], List[str]]:
        """
        Process a single row to tokenize the stringified tokens into sentences
        and return the lists of tokens and list of ner tags for each sentence.
        """
        tokens = row['tokens']
        ner_tags = [self.id2label[tag] for tag in row['ner_tags']]

        # Assuming sentences are already tokenized and separated in 'tokens'
        return tokens, ner_tags

    def generate_similar_pairs_for_row(self, tokens, ner_tags, all_pairs):
        """
        Generate all similar sentence pairs for a given row in the dataset.
        :param tokens: List of tokens from the row.
        :param ner_tags: List of NER tags from the row.
        :param all_pairs: The list where the generated pairs will be added.
        """
        entity_sentences = self.create_entity_based_sentences(tokens, ner_tags)
        sentence = " ".join(tokens)

        for entity_sentence in entity_sentences:
            similar_pair = {
                'sentence1': sentence,
                'sentence2': entity_sentence,
                'label': 'similar'
            }
            all_pairs.append(similar_pair)

    def generate_dissimilar_pairs_for_row(self, tokens, ner_tags, all_pairs, all_entities):
        """
        Generate dissimilar sentence pairs for a given row in the dataset.
        :param tokens: List of tokens from the row.
        :param ner_tags: List of NER tags from the row.
        :param all_pairs: The list where the generated pairs will be added.
        :param all_entities: A set containing all unique entities across the dataset.
        """
        sentence = " ".join(tokens)
        row_entities = set(self.create_entity_based_sentences(tokens, ner_tags))

        for entity in all_entities - row_entities:
            dissimilar_pair = {
                'sentence1': sentence,
                'sentence2': entity,
                'label': 'not_similar'
            }
            all_pairs.append(dissimilar_pair)

    def create_entity_based_sentences(self, tokens, ner_tags):
        """
        Create sentences based on detected entities.
        """
        entity_sentences = []
        unique_entities = set(ner_tags) - {'O'}

        for entity in unique_entities:
            entity_sentence = entity.replace("B-", "").replace("I-", "")
            entity_sentences.append(entity_sentence)

        return entity_sentences

    def add_entity_based_pairs(self, sentence, entity_sentences, all_pairs):
        """
        Add pairs of sentences and entity-based sentences to the list of pairs.
        """
        for entity_sentence in entity_sentences:
            similar_pair = {
                'sentence1': sentence,
                'sentence2': entity_sentence,
                'label': 'similar'
            }
            all_pairs.append(similar_pair)

    def create_sentence_pairs(self, dataset):
        """
        Create all the similar/dissimilar sentence pairs for each sentence in the dataset.
        """
        all_sentence_pairs = []

        for row1, row2 in itertools.combinations(dataset, 2):
            tokens1, ner_tags1 = self.process_row(row1)
            tokens2, ner_tags2 = self.process_row(row2)

            sentence1 = " ".join(tokens1)
            sentence2 = " ".join(tokens2)

            is_similar = any(tag1 == tag2 and tag1 != 'O' for tag1, tag2 in zip(ner_tags1, ner_tags2))
            pair = {'sentence1': sentence1, 'sentence2': sentence2, 'label': 'similar' if is_similar else 'not_similar'}
            all_sentence_pairs.append(pair)

            # Handle special cases
            entity_sentences1 = self.create_entity_based_sentences(tokens1, ner_tags1)
            entity_sentences2 = self.create_entity_based_sentences(tokens2, ner_tags2)

            self.add_entity_based_pairs(sentence1, entity_sentences1, all_sentence_pairs)
            self.add_entity_based_pairs(sentence2, entity_sentences2, all_sentence_pairs)

        return all_sentence_pairs

    def create_dissimilar_entity_pairs(self, all_sentence_pairs):
        """
        Create dissimilar entity pairs from the sentence pairs.
        """
        for pair1, pair2 in itertools.combinations(all_sentence_pairs, 2):
            if pair1['label'] == pair2['label'] == 'similar':
                continue  # Skip if both pairs are similar

            dissimilar_pair = {
                'sentence1': pair1['sentence1'],
                'sentence2': pair2['sentence2'],
                'label': 'not_similar'
            }
            all_sentence_pairs.append(dissimilar_pair)

    def generate_random_dissimilar_pairs(self, dataset, num_pairs: int):
        """
        Generate a specified number of random dissimilar sentence pairs from the dataset.
        :param dataset: The dataset to generate pairs from.
        :param num_pairs: The number of dissimilar pairs to generate.
        """
        pass

    def process_dataset(self, dataset):
        sentence_pairs = self.create_sentence_pairs(dataset)
        self.create_dissimilar_entity_pairs(sentence_pairs)  # Adding dissimilar pairs
        return sentence_pairs


# Create an instance of the class
dataset_constructor = NERSiameseDataSetConstructor()

# Here is the path to the dataset that we want to load from disk.
dataset_path = r"C:\Users\doren\PycharmProjects\GANTRITHOR_FINAL_2024\TESTCODE\FineTuningModel\saved_dataset_8f6662f3d4"

# Define a path where you want to save the processed Siamese dataset
save_directory = r"C:\Users\doren\PycharmProjects\GANTRITHOR_FINAL_2024\TESTCODE\FineTuningModel"

# Run the Siamese dataset construction pipeline
dataset_constructor.run_siamese_dataset_construction_pipeline(directory=dataset_path, save_directory=save_directory)


