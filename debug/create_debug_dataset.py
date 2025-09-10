from pathlib import Path

import pandas as pd

print("Creating STRATIFIED debug datasets...")

data_dir = Path("data")
train_path = data_dir / "train.csv"
dev_path = data_dir / "dev_full.csv"

train_debug_path = data_dir / "train_debug.csv"
dev_debug_path = data_dir / "dev_debug.csv"

NUM_SAMPLES_PER_CLASS = 20
NUM_DEV_SAMPLES = 20

try:
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)

    train_debug_df = train_df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(min(len(x), NUM_SAMPLES_PER_CLASS)),
    )
    train_debug_df = train_debug_df.sample(frac=1).reset_index(drop=True)

    dev_debug_df = dev_df.head(NUM_DEV_SAMPLES)

    train_debug_df.to_csv(train_debug_path, index=False)
    dev_debug_df.to_csv(dev_debug_path, index=False)

    print(f"Successfully created '{train_debug_path}' with {len(train_debug_df)} rows.")
    print(
        f"Train debug dataset label distribution:\n{train_debug_df['label'].value_counts()}",
    )
    print(f"\nSuccessfully created '{dev_debug_path}' with {len(dev_debug_df)} rows.")
    print("\nStratified debug dataset is ready!")

except FileNotFoundError as e:
    print(f"\nERROR: Original data file not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
