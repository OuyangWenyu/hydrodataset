"""Read data from multiple specified CAMELS datasets.

This script demonstrates using MultiDatasetReader to collect IDs and read data
from specified datasets.
"""

from hydrodataset import StandardVariable
from hydrodataset.multi_dataset_reader import MultiDatasetReader, DATASET_MAPPING


def main():
    """Main function for reading specified datasets."""
    # Option 1: Specify the datasets to use (currently active)
    # DATASETS = [
    #     "camels_us",
    #     "camels_cl",
    #     "camels_col",
    #     "camels_dk",
    #     "camels_fi",
    #     "camels_gb",
    #     "camels_lux",
    # ]

    # Option 2: Use ALL datasets from DATASET_MAPPING (47 datasets)
    # Uncomment the line below to read all available datasets
    DATASETS = list(DATASET_MAPPING.keys())

    print("=" * 80)
    print("Multi-Dataset Reader")
    print("=" * 80)
    print(f"Specified datasets ({len(DATASETS)}):")
    for ds in DATASETS:
        print(f"  - {ds}")
    print()

    # Initialize MultiDatasetReader with specified datasets
    print("Initializing MultiDatasetReader...")
    reader = MultiDatasetReader(datasets=DATASETS)

    # Step 1: Collect all station IDs from specified datasets
    print("\n" + "=" * 80)
    print("Step 1: Collecting Station IDs")
    print("=" * 80)
    id_mapping = reader.collect_all_ids(force_refresh=False)

    # Display ID statistics
    print("\nID Collection Results:")
    print("-" * 80)
    print(f"{'Dataset':<20} {'Stations':>10}")
    print("-" * 80)
    total_ids = 0
    for dataset_name in DATASETS:
        count = len(id_mapping.get(dataset_name, []))
        total_ids += count
        print(f"{dataset_name:<20} {count:>10,}")
    print("-" * 80)
    print(f"{'TOTAL':<20} {total_ids:>10,}")

    # Step 2: Check for duplicate IDs across datasets
    print("\n" + "=" * 80)
    print("Step 2: Checking for Duplicate IDs")
    print("=" * 80)
    unique_ids, duplicate_dict = reader.get_global_unique_ids(id_mapping)
    duplicates_count = total_ids - len(unique_ids)
    print(f"Total IDs collected: {total_ids:,}")
    print(f"Globally unique IDs: {len(unique_ids):,}")

    if duplicates_count > 0:
        print(f"Duplicate IDs: {duplicates_count:,}")

        # Display detailed duplicate information
        print("\n" + "-" * 80)
        print("Detailed Duplicate ID Information:")
        print("-" * 80)
        print(f"{'Duplicate ID':<20} {'Count':>8} {'Appears in Datasets'}")
        print("-" * 80)

        # Sort by number of occurrences (descending)
        sorted_duplicates = sorted(
            duplicate_dict.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        # Show all duplicates (or limit to first 20 if too many)
        display_limit = 20
        for i, (dup_id, datasets) in enumerate(sorted_duplicates):
            if i >= display_limit:
                remaining = len(sorted_duplicates) - display_limit
                print(f"... and {remaining} more duplicate IDs")
                break

            datasets_str = ", ".join(datasets)
            if len(datasets_str) > 50:
                datasets_str = datasets_str[:47] + "..."

            print(f"{dup_id:<20} {len(datasets):>8} {datasets_str}")

        print("-" * 80)

        # Summary statistics by dataset
        print("\nDuplicate ID Statistics by Dataset:")
        print("-" * 80)
        print(f"{'Dataset':<20} {'IDs with Duplicates':>20}")
        print("-" * 80)

        dataset_dup_count = {}
        for dup_id, datasets in duplicate_dict.items():
            for dataset in datasets:
                dataset_dup_count[dataset] = dataset_dup_count.get(dataset, 0) + 1

        for dataset_name in sorted(dataset_dup_count.keys()):
            count = dataset_dup_count[dataset_name]
            print(f"{dataset_name:<20} {count:>20,}")

        print("-" * 80)
    else:
        print("No duplicates found!")

    # Step 3: Read duplicate IDs data (if any)
    if duplicates_count > 0:
        print("\n" + "=" * 80)
        print("Step 3: Reading Data for Duplicate IDs")
        print("=" * 80)

        # Get first few duplicate IDs to demonstrate
        sample_duplicate_ids = list(duplicate_dict.keys())[:3]
        print(f"Reading data for {len(sample_duplicate_ids)} duplicate IDs:")
        for dup_id in sample_duplicate_ids:
            datasets_str = ", ".join(duplicate_dict[dup_id])
            print(f"  - {dup_id} (appears in: {datasets_str})")

        try:
            duplicate_data = reader.read_data(
                gage_ids=sample_duplicate_ids,
                t_range=["1980-01-01", "1980-12-31"],
                variables=[
                    StandardVariable.STREAMFLOW,
                    StandardVariable.PRECIPITATION,
                ],
                include_duplicates=True,  # Read from all datasets
            )

            # Display results
            print("\n" + "-" * 80)
            print("Duplicate ID Data Summary:")
            print("-" * 80)
            print(f"Total data entries read: {len(duplicate_data)}")

            # Group by original ID
            for dup_id in sample_duplicate_ids:
                print(f"\nID '{dup_id}':")
                matching_keys = [k for k in duplicate_data.keys() if k.startswith(f"{dup_id}@")]
                for key in matching_keys:
                    dataset_name = key.split("@")[1]
                    df = duplicate_data[key]
                    print(f"  - From {dataset_name}: {df.shape}")
                    print(f"    Columns: {list(df.columns)}")
                    print(f"    Date range: {df.index.min()} to {df.index.max()}")

                    # Show first few rows
                    print(f"    First 3 rows:")
                    print(df.head(3).to_string(index=True))

        except Exception as e:
            print(f"\nError reading duplicate data: {e}")
            import traceback
            traceback.print_exc()

    # Step 4: Read sample data (non-duplicate IDs)
    print("\n" + "=" * 80)
    print("Step 4: Reading Sample Data (Non-duplicate IDs)")
    print("=" * 80)

    # ========== Configuration: Specify station IDs to read ==========
    # Option 1: Set to None to automatically select first N stations
    # Option 2: Set to a list of IDs to manually specify stations
    ids_to_read = [
        "01013500",  # St. John River at Ninemile Bridge, Maine
        "01022500",  # Narraguagus River at Cherryfield, Maine
        "01030500",  # Mattawamkeag River near Mattawamkeag, Maine
    ]
    # To use automatic selection, uncomment the line below:
    # ids_to_read = None
    # ================================================================

    # Determine station IDs based on configuration
    if ids_to_read is None:
        # Automatic mode: select first 3 stations from first available dataset
        source_dataset = None
        for dataset_name in DATASETS:
            if id_mapping.get(dataset_name):
                ids_to_read = id_mapping[dataset_name][:3]  # Change number to select more/fewer
                source_dataset = dataset_name
                break

        if not ids_to_read:
            print("No IDs available to read data")
            return
        print(f"Auto-selected {len(ids_to_read)} stations from {source_dataset}")
    else:
        # Manual mode: use specified IDs
        source_dataset = "Manually specified"
        print(f"Using {len(ids_to_read)} manually specified stations")

    print(f"Reading data for {len(ids_to_read)} stations from {source_dataset}")
    print(f"Station IDs: {ids_to_read}")
    print(f"Time range: 1980-01-01 to 1980-12-31")
    print(f"Variables: streamflow, precipitation, temperature_mean")

    try:
        data = reader.read_data(
            gage_ids=ids_to_read,
            t_range=["1980-01-01", "1980-12-31"],
            variables=[
                StandardVariable.STREAMFLOW,
                StandardVariable.PRECIPITATION,
                
            ],
        )

        # Display results
        print("\n" + "=" * 80)
        print("Data Reading Summary")
        print("=" * 80)
        print(f"Successfully read data for {len(data)} stations")

        if data:
            # Show details for first station
            first_id = list(data.keys())[0]
            df = data[first_id]

            print(f"\n" + "-" * 80)
            print(f"Sample Data - Station ID: {first_id}")
            print("-" * 80)
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Columns: {list(df.columns)}")

            print(f"\nFirst 10 rows:")
            print(df.head(10))

            print("\n" + "=" * 80)
            print("Process Complete")
            print("=" * 80)

    except Exception as e:
        print(f"\nError reading data: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
