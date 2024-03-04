
from datasets import concatenate_datasets, ClassLabel, DatasetDict



class DatasetMerge:
    def __init__(self, datasets):
        self.datasets = datasets

    """
    For this class, it will essentially do the opposite of what DatasetSplit did.

    This class will take a list of datasets,
    and each dataset past the first one must have the exact same Features structure (same datatypes, i.e, Sequence, Value, dtype, id, ect).
    The specified class-labels, however, are allowed to differ.

    So, the above description will be our first method.

    Now, given that all the datasets are the same features type. We will want to pool all the ClassLabels together.
    We will also have to relabel and realign the id2labels, if that makes sense.

    So for example if one dataset has ClassLabels: O, ORG, PER and the other has classLabels O, MISC, LOC
    then id2label in class one is 0: O, 1: ORG, 2:PER
    and id2label in class two is 0: O, 1: MISC, 2: LOC

    we will want to end up, after merging, with something like this:

    0: O, 1: ORG, 2:PER 3: MISC, 4: LOC.

    Understand?

    Tell me how to accomplish this.


    """

    def merge(self):
        # Verify that all datasets have the same feature structure as the first dataset
        first_dataset_features = self.datasets[0].features
        for dataset in self.datasets[1:]:
            if dataset.features != first_dataset_features:
                raise ValueError("All datasets must have the same feature structure as the first dataset")

        # Concatenate the datasets
        merged_dataset = concatenate_datasets(self.datasets)

        # Merge ClassLabels and reassign labels
        if 'label' in merged_dataset.features and isinstance(merged_dataset.features['label'], ClassLabel):
            all_labels = set()
            for dataset in self.datasets:
                all_labels.update(dataset.features['label'].names)

            # Create a new ClassLabel feature with the merged labels
            new_class_label = ClassLabel(names=sorted(all_labels))

            # Update the labels in the merged dataset
            merged_dataset = merged_dataset.map(lambda example: {'label': new_class_label.str2int(example['label'])})

        return merged_dataset
