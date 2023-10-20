import os
import csv
import argparse


def replace_path_in_csv(csv_file, old_path, new_path):
    """
    Replace the old_path with new_path in the given csv file.
    """
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Modify the path in the csv content
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if old_path in cell:
                rows[i][j] = cell.replace(old_path, new_path)

    # Write the modified content back to csv
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Update path in CSV files.")
    parser.add_argument(
        "dir_path", help="Path to the directory containing CSV files.")
    args = parser.parse_args()

    old_path = "H:\\Projects\\C1308_SEAT\\SEAT_inputs\\style_files"

    for filename in os.listdir(args.dir_path):
        if filename.endswith('.csv'):
            full_file_path = os.path.join(args.dir_path, filename)
            replace_path_in_csv(full_file_path, old_path, args.dir_path)

    print(
        f"Paths in CSV files in directory {args.dir_path} have been updated.")
