from datasets import load_dataset




from datasets import load_dataset

class DatasetSplit:
    """
    We want this for text classification and named entity recognition.
    For text classification, its quite simple.

    However for NER, we have the issue that, some rows will have multiple ner_tags (labels) in them.
    We will want to make this class detect, based off features, what task the dataset is built for.

    Then, if its built for ner_tags, we will want the following to happen:
    By default, if two rows have two non 'O' labels that are in both lists, then the row will be placed into both datasets.
    Call this parameter: exclusion=False.

    If the user makes exclusion=True, then we want to prevent either row from being in either dataset.

    TODO: rewrite this class with the added functionality described above.


    """

    def __init__(self, dataset_name, split_labels_1, split_labels_2, exclusion=False):
        self.dataset_name = dataset_name
        self.split_labels_1 = set(split_labels_1)
        self.split_labels_2 = set(split_labels_2)
        self.exclusion = exclusion

    def split(self):
        # Load the dataset
        dataset = load_dataset(self.dataset_name)

        # Determine if the dataset is for NER based on its features
        is_ner = 'ner_tags' in dataset.features

        if is_ner:
            # Split the dataset for NER
            def filter_fn(example):
                labels_1 = set(example['ner_tags']) & self.split_labels_1
                labels_2 = set(example['ner_tags']) & self.split_labels_2
                if self.exclusion:
                    return bool(labels_1) != bool(labels_2)
                else:
                    return bool(labels_1) or bool(labels_2)

            dataset_1 = dataset.filter(lambda example: set(example['ner_tags']) & self.split_labels_1)
            dataset_2 = dataset.filter(lambda example: set(example['ner_tags']) & self.split_labels_2)
        else:
            # Split the dataset for text classification
            dataset_1 = dataset.filter(lambda example: example['label'] in self.split_labels_1)
            dataset_2 = dataset.filter(lambda example: example['label'] in self.split_labels_2)

        return dataset_1, dataset_2

# Example usage for text classification
splitter = DatasetSplit('imdb', split_labels_1=[0], split_labels_2=[1])
dataset_1, dataset_2 = splitter.split()

# Example usage for NER with exclusion
splitter_ner = DatasetSplit('conll2003', split_labels_1=[1, 2], split_labels_2=[3, 4], exclusion=True)
dataset_1_ner, dataset_2_ner = splitter_ner.split()
