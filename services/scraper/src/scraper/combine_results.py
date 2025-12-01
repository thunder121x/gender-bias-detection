import os
import glob
from pathlib import Path
import pandas as pd

from scraper.constants import OUTPUT_DIR

def combine_all_csv(base_dir=OUTPUT_DIR, output_path=Path(OUTPUT_DIR,"combined_output.csv")):
    """
    Recursively read all CSV files under the base directory
    and combine them assuming they share the same schema.
    """

    # Recursively find CSVs
    csv_files = glob.glob(os.path.join(base_dir, "**/*.csv"), recursive=True)

    print(f"Found {len(csv_files)} CSV files.")

    if len(csv_files) == 0:
        print("No CSV files found.")
        return

    df_list = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = file       # (Optional) Track where each row came from
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Combine into one dataframe
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save final file
    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Combined CSV {combined_df.shape} saved to: {output_path}")
    return combined_df

if __name__ == "__main__":
    combine_all_csv()