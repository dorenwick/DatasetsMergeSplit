import logging
import os
import traceback
from datasets import load_from_disk, load_dataset, Value

class DatasetToTextFilesConverter:
    def __init__(self, dataset_paths, output_dir, lines_per_file=500, max_line_length=1500, slice_size=10000):
        self.dataset_paths = dataset_paths
        self.output_dir = output_dir
        self.lines_per_file = lines_per_file
        self.max_line_length = max_line_length
        self.slice_size = slice_size

    def convert_datasets_to_text_files(self):
        for dataset_path in self.dataset_paths:
            self._create_text_files_from_dataset(dataset_path)

    def _create_text_files_from_dataset(self, dataset_path):
        # Load the dataset
        try:
            dataset = load_from_disk(dataset_path)
        except Exception as e:
            print(f"Error loading dataset from disk: {e}. Attempting to load from Hugging Face Hub instead...")
            try:
                dataset = load_dataset(dataset_path)
            except Exception as e:
                logging.error(f"Error loading dataset from Hugging Face Hub: {e}")
                traceback.print_exc()
                return

        print(f"Dataset loaded: {dataset_path}")
        print(f"Features: {dataset['train'].features}")

        text_feature_name = self._find_text_feature(dataset['train'].features)
        if not text_feature_name:
            print("No text feature found in the dataset.")
            return

        print(f"Text feature found: {text_feature_name}")

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Iterate over the dataset in slices and write to text files
        file_index = 0
        line_count = 0
        current_file = open(os.path.join(self.output_dir, f"output_{file_index}.txt"), "w", encoding="utf-8")

        for text_content in dataset["train"][text_feature_name]:  # Corrected iteration
            if not text_content:
                continue

            # Write text content to the file
            current_file.write(text_content + "\n")
            line_count += 1

            if line_count >= self.lines_per_file:
                current_file.close()
                file_index += 1
                line_count = 0
                current_file = open(os.path.join(self.output_dir, f"output_{file_index}.txt"), "w", encoding="utf-8")

        # Close the last file
        current_file.close()

    def _find_text_feature(self, features):
        # Find the feature that is likely to be the text content
        for feature_name, feature_type in features.items():
            print(f"Checking feature: {feature_name}, type: {feature_type}")
            if isinstance(feature_type, Value) and feature_type.dtype == 'string':
                print(f"Found text feature: {feature_name}")
                return feature_name

        # Fallback: return the first feature name that contains "string" in its type description
        for feature_name, feature_type in features.items():
            if "string" in str(feature_type):
                print(f"Fallback - found text feature: {feature_name}")
                return feature_name

        print("No text feature found.")
        return None


# Example usage
dataset_paths = [
    r"C:\Users\doren\.cache\huggingface\datasets\tanay___sentiment-corpus\default\0.0.0\24e9f1ae526fa2c4",
]
output_dir = r"C:\Users\doren\AppData\Roaming\Gantrithor\data\datasets\OutputTextFilesSentiment"
converter = DatasetToTextFilesConverter(dataset_paths, output_dir, slice_size=50000)
converter.convert_datasets_to_text_files()
