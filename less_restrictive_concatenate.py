from datasets import Dataset, Features, Sequence, Value, ClassLabel
import pandas as pd
import itertools

def are_datasets_compatible(dataset1, dataset2):
    if len(dataset1.features) != len(dataset2.features):
        return False
    for feature_value in dataset1.features.values():
        compatible_feature_found = False
        for other_feature in dataset2.features.values():
            if isinstance(feature_value, Sequence) and isinstance(feature_value.feature, ClassLabel) and \
                    isinstance(other_feature, Sequence) and isinstance(other_feature.feature, ClassLabel):
                compatible_feature_found = True
                break
            elif type(feature_value) == type(other_feature):
                compatible_feature_found = True
                break
        if not compatible_feature_found:
            return False
    return True

def merge_dataset_test(datasets):
    """
    Checks if a list of datasets can be merged based on their schemas.
    All feature schemas must be identical except for the ClassLabel names and feature keys.

    We do not care about anything in class labels names except that they are both lists of elements.
    Length difference doesn't matter, for example. That's fine, and should not cause an invalid break for us.

    For every schema in one dataset, this function checks if a compatible schema exists in the other datasets
    (outside of ClassLabel names and feature keys). If no such schema is found, it prints 'Invalid dataset' and removes the dataset
    from the final dataset list. Otherwise, it prints 'Valid dataset'.

    Here is a more indepth explanation:

    we will use these two dataset FeatureSchemas as example:

    mixed_features = Features({
        "text": Sequence(feature=Value(dtype="string")),
        "labels": Sequence(feature=ClassLabel(names=["label1", "label2"])),
        "id": Value(dtype="int32")
    })

    same_features = Features({
        "text": Sequence(feature=Value(dtype="string")),
        "category": Sequence(feature=ClassLabel(names=["cat1", "cat2", "dog"])),
        "index": Value(dtype="int32")
    })

    We want to check if two datasets can be merged:
    Now, for the features schema of two datasets, we want to check the following:
    If every feature column in a dataset has a matching feature column in the other dataset for the values.
    We do not need the key names to match.

    For example for "id": Value(dtype="int32") in mixed_features, we will find "index": Value(dtype="int32").
    The keys here do not match, but the values in this item do. So that's valid.
    We are looking for datasets where each value of the schema has an identical value in the other dataset (with an exception
    that we will discuss).

    So, key names do not matter if they differ.

    Now, there is an exception.

    For any Features schema that has feature=Classlabel in it, we will allow for their names to differ.
    So for example, given
    "labels": Sequence(feature=ClassLabel(names=["label1", "label2"])),
    "category": Sequence(feature=ClassLabel(names=["cat1", "cat2", "dog"])),

    we only care that this part is the same:
    Sequence(feature=ClassLabel(names=[...])),
    and so we will allow for a match even though
    names=["label1", "label2"] != names=["cat1", "cat2", "dog"]
    All we want to check is that the basic structure here is correct, and the names being different should not make the datasets
    invalid.

    So, to summarize:
    Datasets are valid if and only if all the values in the Features schema have a matching value in the other dataset,
    except for values that contain any ClassLabels, where the names are allowed to differ.

    """

    num_datasets = len(datasets)
    compatibility_matrix = [[False] * num_datasets for _ in range(num_datasets)]

    for i, j in itertools.combinations(range(num_datasets), 2):
        dataset1, dataset2 = datasets[i], datasets[j]
        if are_datasets_compatible(dataset1[0], dataset2[0]):
            compatibility_matrix[i][j] = True
            compatibility_matrix[j][i] = True

    equivalence_classes = []
    visited = [False] * num_datasets

    for i in range(num_datasets):
        if not visited[i]:
            equivalence_class = [datasets[i]]
            visited[i] = True
            for j in range(i + 1, num_datasets):
                if compatibility_matrix[i][j]:
                    equivalence_class.append(datasets[j])
                    visited[j] = True
            equivalence_classes.append(equivalence_class)

    print(f"Equivalence classes of compatible datasets:")
    for i, equivalence_class in enumerate(equivalence_classes):
        print(f"Class {i + 1}: {[dataset[1] for dataset in equivalence_class]}")

    return equivalence_classes


# Test the function
if __name__ == "__main__":
    # Create sample datasets
    mixed_features = Features({
        "text": Sequence(feature=Value(dtype="string")),
        "labels": Sequence(feature=ClassLabel(names=["label1", "label2"])),
        "id": Value(dtype="int32")
    })
    same_features = Features({
        "text": Sequence(feature=Value(dtype="string")),
        "category": Sequence(feature=ClassLabel(names=["cat1", "cat2", "dog"])),
        "index": Value(dtype="int32")
    })
    single_feature_features = Features({
        "text": Value(dtype="string"),
        "id": Value(dtype="int32")
    })
    different_features = Features({
        "sentence": Value(dtype="string"),
        "category": ClassLabel(names=["cat1", "cat2"]),
        "index": Value(dtype="int32")
    })

    mixed_data = {
        "text": [["This", "is", "a", "sentence"], ["Another", "sentence"]],
        "labels": [[0, 1], [1, 0]],
        "id": [1, 2]
    }
    same_data = {
        "text": [["This", "is", "a", "sentence"], ["Another", "sentence"]],
        "category": [[0, 1, 2], [2, 0, 1]],
        "index": [1, 2]
    }
    single_feature_data = {
        "text": ["This is a sentence.", "Another sentence."],
        "id": [1, 2]
    }
    different_data = {
        "sentence": ["This is a different sentence.", "Yet another different sentence."],
        "category": [0, 1],
        "index": [1, 2]
    }

    mixed_dataset = (Dataset.from_dict(mixed_data, features=mixed_features), "mixed_dataset")
    single_feature_dataset = (Dataset.from_dict(single_feature_data, features=single_feature_features), "single_feature_dataset")
    different_dataset = (Dataset.from_dict(different_data, features=different_features), "different_dataset")
    same_dataset = (Dataset.from_dict(same_data, features=same_features), "same_dataset")

    equivalence_classes = merge_dataset_test([mixed_dataset, same_dataset, single_feature_dataset, different_dataset])
    print(f"Number of equivalence classes: {len(equivalence_classes)}")