from datasets import concatenate_datasets, ClassLabel, DatasetDict, load_from_disk, Sequence


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

class DatasetMerge:
    def __init__(self, datasets):
        self.datasets = datasets

    def merge(self):
        # Verify that all datasets have the same feature structure as the first dataset (ignoring class labels)
        first_dataset_features = self._get_features(self.datasets[0], ignore_labels=True)
        for dataset in self.datasets[1:]:
            if self._get_features(dataset, ignore_labels=True) != first_dataset_features:
                raise ValueError("All datasets must have the same feature structure as the first dataset (ignoring class labels)")

        # Prepare datasets for concatenation by updating class labels
        updated_datasets = []
        all_labels = set()
        for dataset in self.datasets:
            all_labels.update(self._get_features(dataset)['ner_tags'].feature.names)

        new_class_label = ClassLabel(names=sorted(all_labels), num_classes=len(all_labels))
        for dataset in self.datasets:
            updated_dataset = dataset.map(lambda example: update_ner_tags(example, new_class_label))
            # Update the ner_tags feature in the dataset's features
            updated_features = updated_dataset.features
            updated_features['ner_tags'] = Sequence(feature=new_class_label)
            updated_dataset = updated_dataset.cast(updated_features)
            updated_datasets.append(updated_dataset)

        # Concatenate the datasets
        merged_dataset = concatenate_datasets(updated_datasets)

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

# Load the datasets
dataset_one = load_from_disk("C:/Users/doren/AppData/Roaming/Gantrithor/data/datasets/saved_dataset_one")['train']
dataset_two = load_from_disk("C:/Users/doren/AppData/Roaming/Gantrithor/data/datasets/saved_dataset_two")['train']

# Create a list of the datasets
datasets = [dataset_one, dataset_two]

# Create an instance of the DatasetMerge class
merger = DatasetMerge(datasets)

# Merge the datasets
merged_dataset = merger.merge()

# Print some information about the merged dataset
print("Number of examples in the merged dataset:", len(merged_dataset))
print("Features of the merged dataset:", merged_dataset.features)
