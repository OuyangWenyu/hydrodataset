"""
Demo: MultiDatasetReader — cross-dataset data access.

Run from project root:
    uv run python examples/read_multi_datasets.py
"""

from hydrodataset.multi_dataset_reader import MultiDatasetReader
from hydrodataset import StandardVariable

# 1. Initialize — specify which datasets to search
reader = MultiDatasetReader(datasets=["camels_us", "camelsh"])
print(f"Datasets: {reader.datasets}")
print(f"Cache:   {reader.cache_dir}")

# 2. Collect all station IDs (cached automatically)
print("\n=== Collect IDs ===")
id_mapping = reader.collect_all_ids()
for ds, ids in id_mapping.items():
    print(f"  {ds}: {len(ids)} stations")

# 3. Cross-dataset deduplication
# Station "01013500" appears in BOTH camels_us and camelsh (~662 stations overlap)
print("\n=== Deduplication ===")
unique_ids, duplicates = reader.get_global_unique_ids()
print(f"  Unique IDs: {len(unique_ids)}")
print(f"  Duplicate IDs: {len(duplicates)}")
print(f"  Example: 01013500 appears in {duplicates.get('01013500', [])}")

# 4. Read a duplicate station — output from BOTH datasets
print("\n=== Read duplicate station 01013500 ===")
data = reader.read_data(
    gage_ids=["01013500"],
    t_range=["2000-01-01", "2000-01-10"],
    variables=[StandardVariable.STREAMFLOW],
)
for key, df in data.items():
    print(f"  {key}: {df.shape}")
    print(df.head(2))

# 5. S3 mode (requires cloud credentials)
# reader_cloud = MultiDatasetReader(datasets=["camels_us"], source="cloud")

print("\nDone.")
