from netCDF4 import Dataset
import numpy as np

# Define the paths to the datasets
datasets_paths = [
    'tests/data/unstructured/mec-not-present/downsampled_0_tanana_1_map.nc',
    'tests/data/unstructured/mec-not-present/downsampled_0_tanana_100_map.nc',
    'tests/data/unstructured/mec-present/downsampled_9_tanana_1_map.nc',
    'tests/data/unstructured/mec-present/downsampled_9_tanana_100_map.nc'
]

# Loop through each dataset and print information
for idx, dataset_path in enumerate(datasets_paths, 1):
    dataset = Dataset(dataset_path, 'r')

    # print(f"\nDataset {idx}: {dataset_path}")

    # # Print the variable keys
    # print("\nVariables:")
    # print(list(dataset.variables.keys()))

    # # Print the dimensions and their sizes
    # print("\nDimensions:")
    # for dim in dataset.dimensions.items():
    #     print(f"{dim[0]}: {len(dim[1])}")

    # # Print global attributes of the dataset
    # print("\nGlobal Attributes:")
    # for attr in dataset.ncattrs():
    #     print(f"{attr}: {getattr(dataset, attr)}")

    # # Extract and analyze data for key variables
    # for var_name in ['ucxa', 'ucya', 'unorm', 'waterdepth', 'motility_classified']:
    #     if var_name in dataset.variables:
    #         var_data = dataset.variables[var_name][:]

    #         # Check for valid (non-NaN and non-zero) values
    #         if var_name == 'motility_classified':
    #             valid_data = var_data[~np.isnan(var_data) & (var_data != -100)]
    #         else:
    #             valid_data = var_data[~np.isnan(var_data) & (var_data != 0)]

    #         if valid_data.size > 0:
    #             # Print statistics about the variable
    #             print(f"\nStatistics for '{var_name}':")
    #             print(f"Min: {np.nanmin(valid_data)}")
    #             print(f"Max: {np.nanmax(valid_data)}")
    #             print(f"Mean: {np.nanmean(valid_data)}")
    #             print(f"Standard Deviation: {np.nanstd(valid_data)}")

    #             # Optional: print some sample values
    #             print(f"\nSample '{var_name}' data points:")
    #             print(valid_data[:5])  # Print first 5 valid data points
    #         else:
    #             print(f"\n'{var_name}' contains no valid (non-NaN, non-placeholder) data.")

    # Analyze the distribution of special values in 'motility_classified'

    if 'motility_classified' in dataset.variables:
        motility_data = dataset.variables['motility_classified'][:]
        negative_count = np.sum(motility_data == -100)
        total_count = motility_data.size

        print(f"\n'motility_classified' Special Value Analysis:")
        print(f"Number of -100 values: {negative_count}")
        print(f"Total data points: {total_count}")
        print(f"Percentage of -100 values: {100 * negative_count / total_count:.2f}%")

    # Close the dataset
    dataset.close()
