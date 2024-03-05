import pandas as pd
from datasets import concatenate_datasets, DatasetDict, load_from_disk, Sequence, Value, ClassLabel, Features
import os
import pyarrow.parquet as pq
from datasets import Dataset
from datasets import Features, ClassLabel, Sequence, Value


class DatasetMerge:
    """



    """


    def __init__(self, datasets, output_dir):
        self.datasets = datasets
        self.output_dir = output_dir
        self.label_indices = []  # List to store label index mappings for each dataset

    def create_label_index_mappings(self):
        self.all_labels = set()
        new_label_index = 0
        self.all_id2labels = []  # List to store id2labels dictionaries for each dataset

        # Find the intersection of labels and order them by integer encodings
        common_labels = set.intersection(
            *[set(dataset.features['ner_tags'].feature.names) for dataset in self.datasets])
        for label in sorted(common_labels):
            self.all_labels.add(label)
            new_label_index += 1

        # Create label index mappings for each dataset
        for dataset in self.datasets:
            labels = dataset.features['ner_tags'].feature.names
            old_index_to_new_index = {}
            id2labels = {}  # Dictionary to map integers to labels

            # Map common labels to their indices
            for label in common_labels:
                old_index = labels.index(label)
                new_index = sorted(list(common_labels)).index(label)
                old_index_to_new_index[old_index] = new_index
                id2labels[new_index] = label  # Map new index to label

            # Map unique labels to new indices
            unique_labels = [label for label in labels if label not in common_labels]
            for label in unique_labels:
                old_index = labels.index(label)
                new_index = new_label_index
                old_index_to_new_index[old_index] = new_index
                id2labels[new_index] = label  # Map new index to label
                self.all_labels.add(label)
                new_label_index += 1

            self.label_indices.append(old_index_to_new_index)
            self.all_id2labels.append(id2labels)  # Append id2labels dictionary for this dataset

        self.combine_and_sort_id2labels()

    def combine_and_sort_id2labels(self):
        # Combine all_id2labels dictionaries into a single dictionary
        self.id2label = {}
        for id2labels_dict in self.all_id2labels:
            self.id2label.update(id2labels_dict)

        # Sort the combined dictionary by keys
        self.id2label = dict(sorted(self.id2label.items()))

        # Update all_labels with the values from the sorted id2label dictionary
        self.all_labels = list(self.id2label.values())

    def merge(self):
        # Create label index mappings
        self.create_label_index_mappings()

        # Convert datasets to dataframes and update ner_tags
        dataframes = []
        for i, dataset in enumerate(self.datasets):
            df = dataset.to_pandas()
            label_index = self.label_indices[i]

            def update_tags(row):
                return [label_index[tag] for tag in row['ner_tags']]

            df['ner_tags'] = df.apply(update_tags, axis=1)

            # Sort the dataframe by document_id
            df = df.sort_values(by='document_id')

            dataframes.append(df)

        # Concatenate the dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)

        # Reset document_id
        merged_df['document_id'] = range(len(merged_df))

        # Convert the merged dataframe back to a dataset
        merged_dataset = Dataset.from_pandas(merged_df)

        # Update the ner_tags feature to have the same ClassLabel names across all datasets
        # Define the features for the merged dataset
        features = Features({
            "tokens": Sequence(feature=Value(dtype="string")),
            "ner_tags": Sequence(feature=ClassLabel(names=self.all_labels)),
            "document_id": Value(dtype="int32")
        })

        # Cast the merged dataset to the defined features
        merged_dataset = merged_dataset.cast(features)

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
