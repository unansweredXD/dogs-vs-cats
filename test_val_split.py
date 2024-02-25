import os
from sklearn.model_selection import train_test_split
from shutil import move


def split_data(input_dir, output_train_dir, output_val_dir, output_test_dir, train_size=0.6, random_seed=42):
    files = os.listdir(input_dir)
    train_files, temp_files = train_test_split(files, test_size=1 - train_size, random_state=random_seed)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=random_seed)

    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    for file in train_files:
        src_path = os.path.join(input_dir, file)
        dest_path = os.path.join(output_train_dir, file)
        move(src_path, dest_path)

    for file in val_files:
        src_path = os.path.join(input_dir, file)
        dest_path = os.path.join(output_val_dir, file)
        move(src_path, dest_path)

    for file in test_files:
        src_path = os.path.join(input_dir, file)
        dest_path = os.path.join(output_test_dir, file)
        move(src_path, dest_path)


if __name__ == "__main__":
    input_data_dir = "train_all"
    output_train_data_dir = "train"
    output_val_data_dir = "val"
    output_test_data_dir = "test"
    train_split_size = 0.6
    random_seed = 42

    split_data(input_data_dir, output_train_data_dir, output_val_data_dir, output_test_data_dir,
               train_size=train_split_size, random_seed=random_seed)
