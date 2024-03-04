from datasets import concatenate_datasets, DatasetDict, load_from_disk, Sequence, Value, ClassLabel, Features
import os
import pyarrow.parquet as pq



class DatasetMerge:
    """
    Here is what we need to do. We need to create a mapping recording the old ner_tag encodings for the labels.
    Example:
    # Example id2label dictionaries for two different datasets:

    # Dataset 1:
    id2label = {
        0: 'O',
        1: 'B-PER',
        2: 'B-other',
        3: 'B-ORG',
        4: 'I-PER',
        5: 'I-MISC',
        6: 'B-MISC',
        7: 'B-LOC',
        8: 'I-LOC',
        9: 'I-ORG'
    }

    # Dataset 2:
    id2label = {
        0: 'O',
        1: 'B-person',
        2: 'B-product',
        3: 'I-person',
        4: 'I-other',
        5: 'B-location',
        6: 'B-building',
        7: 'I-location',
        8: 'I-event',
        9: 'I-product',
        10: 'I-building',
        11: 'I-organization',
        12: 'I-art',
        13: 'B-art',
        14: 'B-organization',
        15: 'B-event'
    }

    # When merging, we will create a new id2label dictionary that combines labels from both datasets.
    # Overlapping labels will have the same integer encoding.

    # Example merged id2label dictionary:
    id2label = {
        0: 'O',
        1: 'B-PER',
        2: 'B-other',
        3: 'B-ORG',
        4: 'I-PER',
        5: 'I-MISC',
        6: 'B-MISC',
        7: 'B-LOC',
        8: 'I-LOC',
        9: 'I-ORG',
        10: 'B-person',
        11: 'B-product',
        12: 'I-person',
        13: 'I-other',
        14: 'B-location',
        15: 'B-building',
        16: 'I-location',
        17: 'I-event',
        18: 'I-product',
        19: 'I-building',
        20: 'I-organization',
        21: 'I-art',
        22: 'B-art',
        23: 'B-organization',
        24: 'B-event'
    }

    To implement this, we need to map the old ner_tag encodings to the new encodings for each dataset.
    We use the document_id or row index in the dataset to determine which dataset the ner_tag belongs to.

    Example index mappings for each dataset:

    # For Dataset 1:
    Old Index => New Index
    0 => 0 (label 'O')
    5 => 5 (label 'I-MISC')
    9 => 9 (label 'I-ORG')

    # For Dataset 2:
    Old Index => New Index
    0 => 0 (label 'O')
    5 => 14 (label 'B-location')
    15 => 24 (label 'B-event')

    In these examples:
    - For row 117 in dataset 1 with ner_tag 5 ('I-MISC'), the mapping is 5 => 5.
    - For row 2022 in dataset 2 with ner_tag 5 ('B-location'), the mapping is 5 => 14.

    The merge method will implement this system for relabelling, creating a new merged dataset with updated ner_tags.


    TODO: Please write down an index mapping for each ner_tag, for each dataset:
      We will want to use document_id or row index in the dataset to track which dataset we are mapping the ner_tags to.
      So for example if row 117 is in dataset 1 and the ner_tag is 5, then we map 5: 'I-MISC' to 5: 'I-MISC'. So, 5=>5.
      On the other hand if row 2022 is in dataset 2 and the ner_tag is 5, then we map  5: 'B-location' to 14: 'B-location'
      so 5: => 14. Understand?

      Please write some examples of indexes for each dataset to show you understand how the mappings will go.
      Then implement the details.


    # The actual merging logic will handle the creation of this new id2label dictionary.


    When we merge...we find the intersection of labels. and order them by integer encodings.
    We map them to 0,1,2,3...,6 ect if there are 7 of them (as an example)
    Then for the first dataset, we map the new class labels from their encodings to 7,8,9,10
    if there are 4 of them, for eexample.
    And for the next class, map to 11,12,13,...N, as an example.
    So we need to create mappings for all the class labels, from an old integer encoding to a new one.
    Also, we must track which ner_tags are from which dataset, this can probably be done by using document id since we are concatnating the rows of the datasets.
    We need to do that because clearly multiple datasets have the same integer encoding (0,1, for example).

    TODO: Carefully implement this system for relabelling.

    NOTE: Here is an example

    {'B-product', 'B-LOC', 'I-MISC', 'I-PER', 'B-person', 'I-event', 'B-organization', 'B-event', 'I-product', 'I-building', 'I-person', 'I-other', 'B-art', 'B-other', 'I-ORG', 'I-LOC', 'B-MISC', 'B-building', 'I-location', 'I-art', 'B-location', 'I-organization', 'B-PER', 'O', 'B-ORG'}
            [Dataset({
            features: ['tokens', 'ner_tags', 'document_id'],
            num_rows: 1921
        }), Dataset({
            features: ['tokens', 'ner_tags', 'document_id'],
            num_rows: 2803
        })]

    [{'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}, {'O': 0, 'B-art': 9, 'I-art': 10, 'B-building': 11, 'I-building': 12, 'B-event': 13,
     'I-event': 14, 'B-location': 15, 'I-location': 16, 'B-organization': 17, 'I-organization': 18, 'B-other': 19, 'I-other': 20, 'B-person': 21, 'I-person': 22, 'B-product': 23, 'I-product': 24}]

    Ok. So we would do something like this when converting ner_tags.
    First, determine which row the dataset belongs to for the ner_tags. Then, based off this, map the id integer encoding of the ner tag to the string label.
    Then, map the string label



    [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}, {0: 0, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18, 11: 19, 12: 20, 13: 21, 14: 22, 15: 23, 16: 24}]
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
    {0: 0, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18, 11: 19, 12: 20, 13: 21, 14: 22, 15: 23, 16: 24}

    """


    def __init__(self, datasets, output_dir):
        self.datasets = datasets
        self.output_dir = output_dir
        self.label_indices = []  # List to store label index mappings for each dataset

    def create_label_index_mappings(self):
        """
        Creates label index mappings for each dataset and stores them in self.label_indices.
        """
        self.all_labels = set()
        new_label_index = 0

        # Find the intersection of labels and order them by integer encodings
        common_labels = set.intersection(
            *[set(self._get_features(dataset)['ner_tags'].feature.names) for dataset in self.datasets])
        for label in sorted(common_labels):
            self.all_labels.add(label)
            new_label_index += 1

        # Create label index mappings for each dataset
        for i, dataset in enumerate(self.datasets):
            labels = self._get_features(dataset)['ner_tags'].feature.names
            old_index_to_new_index = {}

            # Map common labels to their indices
            for label in common_labels:
                old_index = labels.index(label)
                new_index = sorted(list(common_labels)).index(label)
                old_index_to_new_index[old_index] = new_index

            # Map unique labels to new indices
            unique_labels = [label for label in labels if label not in common_labels]
            for label in unique_labels:
                old_index = labels.index(label)
                new_index = new_label_index
                old_index_to_new_index[old_index] = new_index
                self.all_labels.add(label)
                new_label_index += 1

            self.label_indices.append(old_index_to_new_index)

    def merge(self):
        # Create label index mappings
        self.create_label_index_mappings()

        # Prepare datasets for concatenation by updating class labels
        updated_datasets = []
        cumulative_rows = 0  # Keep track of the cumulative number of rows in the datasets

        for i, dataset in enumerate(self.datasets):
            label_index = self.label_indices[i]
            print(f"Dataset {i} label index mapping: {label_index}")  # Debug print

            def update_example(example, index):
                # Determine which dataset the example belongs to based on its row index
                dataset_index = i if index >= cumulative_rows and index < cumulative_rows + len(dataset) else None
                if dataset_index is not None:
                    try:
                        print("example['ner_tags']  ", example['ner_tags'])
                        for label in example['ner_tags']:
                            print("label  ", label)
                            print("label_index[label]  ", label_index[label])

                        updated_tags = [label_index[label] for label in example['ner_tags']]
                        return {'ner_tags': updated_tags, 'document_id': dataset_index}
                    except KeyError as e:
                        print(f"KeyError for label {e} in dataset {dataset_index}")
                        print(f"ner_tags in the example: {example['ner_tags']}")
                        raise

            updated_dataset = dataset.map(update_example, with_indices=True)

            # Update the ner_tags feature to have the same ClassLabel names across all datasets
            updated_features = updated_dataset.features
            updated_features['ner_tags'] = Sequence(feature=ClassLabel(names=sorted(self.all_labels)))
            updated_dataset = updated_dataset.cast(updated_features)

            updated_datasets.append(updated_dataset)
            cumulative_rows += len(dataset)  # Update the cumulative number of rows

        # Concatenate the datasets
        merged_dataset = concatenate_datasets(updated_datasets)

        # Print the head(10) of the merged dataset
        print("Head of the merged dataset:")
        print(merged_dataset[:10])

        # Print the integer encodings for the ClassLabel feature
        print("Integer encodings for the ClassLabel feature:")
        print(merged_dataset.features['ner_tags'])

        return merged_dataset

    def _get_features(self, dataset, ignore_labels=False):
        # If the dataset is a DatasetDict, use the first split's features
        if isinstance(dataset, DatasetDict):
            features = next(iter(dataset.values())).features
        else:
            features = dataset.features

        # Ignore class labels if requested
        if ignore_labels and 'ner_tags' in features and isinstance(features['ner_tags'].feature, ClassLabel):
            features = features.copy()
            del features['ner_tags']

        return features

    def save_merged_dataset(self, merged_dataset, description="No description provided", citation="No citation provided"):
        # Set dataset info
        merged_dataset.info.description = description
        merged_dataset.info.citation = citation
        merged_dataset.info.builder_name = "custom_dataset_merge"
        merged_dataset.info.config_name = "default"
        merged_dataset.info.version = {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0}

        # Create splits
        dataset_splits = self.create_splits(merged_dataset)
        merged_dataset = dataset_splits

        # Save the dataset
        dataset_dir = self.create_dataset_directory("merged_dataset")
        merged_dataset.save_to_disk(dataset_dir)
        print(f"Dataset saved to {dataset_dir}")

    def create_splits(self, dataset, train_size=0.7, val_size=0.15, test_size=0.15):
        assert train_size + val_size + test_size == 1, "Splits must sum to 1"

        # Split the dataset
        train_test_split = dataset.train_test_split(test_size=test_size + val_size)
        test_val_split = train_test_split['test'].train_test_split(test_size=test_size / (test_size + val_size))

        # Create a DatasetDict
        dataset_splits = DatasetDict({
            'train': train_test_split['train'],
            'validation': test_val_split['train'],
            'test': test_val_split['test']
        })

        return dataset_splits

    def create_dataset_directory(self, dir_name):
        # Create a unique directory name for the dataset
        base_dir = os.path.join(self.output_dir, dir_name)
        dataset_dir = self.create_unique_directory_name(base_dir)

        # Create the directory
        os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir

    def create_unique_directory_name(self, original_path):
        counter = 1
        new_path = f"{original_path}_{counter}"
        while os.path.exists(new_path):
            counter += 1
            new_path = f"{original_path}_{counter}"
        return new_path


def update_ner_tags(example, class_label):
    updated_ner_tags = []
    for label in example['ner_tags']:
        if isinstance(label, int):
            # If the label is already an integer, use it directly
            updated_ner_tags.append(label)
        else:
            # If the label is a string, convert it to an integer
            updated_ner_tags.append(class_label.str2int(label))
    return {'ner_tags': updated_ner_tags}




# Load the datasets
dataset_one = load_from_disk("C:/Users/doren/AppData/Roaming/Gantrithor/data/datasets/saved_dataset_one")['train']
dataset_two = load_from_disk("C:/Users/doren/AppData/Roaming/Gantrithor/data/datasets/saved_dataset_two")['train']

# Create a list of the datasets
datasets = [dataset_one, dataset_two]

save_merge_dataset_path = r"C:/Users/doren/AppData/Roaming/Gantrithor/data/datasets/saved_dataset_merge"

# Create an instance of the DatasetMerge class
merger = DatasetMerge(datasets, save_merge_dataset_path)


merged_dataset = merger.merge()
merger.save_merged_dataset(merged_dataset, description="Merged dataset", citation="No citation provided")

# Print some information about the merged dataset
print("Number of examples in the merged dataset:", len(merged_dataset))
print("Features of the merged dataset:", merged_dataset.features)
