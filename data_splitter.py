import os
from sklearn.model_selection import train_test_split
from shutil import move


class DataSplitter:
    def __init__(
            self,
            input_dir: str = 'train_all',
            output_train_dir: str = 'train',
            output_val_dir: str = 'val',
            output_test_dir: str = 'test',
            train_size: float = 0.6
    ):
        self.input_data_dir = input_dir
        self.output_train_data_dir = output_train_dir
        self.output_val_data_dir = output_val_dir
        self.output_test_data_dir = output_test_dir
        self.train_split_size = train_size

    def split_data(self):
        files = os.listdir(self.input_data_dir)
        train_files, temp_files = train_test_split(files, test_size=1 - self.train_split_size, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        os.makedirs(self.output_train_data_dir, exist_ok=True)
        os.makedirs(self.output_val_data_dir, exist_ok=True)
        os.makedirs(self.output_test_data_dir, exist_ok=True)

        for file in train_files:
            src_path = os.path.join(self.input_data_dir, file)
            destination_path = os.path.join(self.output_train_data_dir, file)
            move(src_path, destination_path)

        for file in val_files:
            src_path = os.path.join(self.input_data_dir, file)
            destination_path = os.path.join(self.output_val_data_dir, file)
            move(src_path, destination_path)

        for file in test_files:
            src_path = os.path.join(self.input_data_dir, file)
            destination_path = os.path.join(self.output_test_data_dir, file)
            move(src_path, destination_path)


if __name__ == "__main__":
    DataSplitter().split_data()
