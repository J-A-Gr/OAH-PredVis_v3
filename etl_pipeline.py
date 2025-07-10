import os
import pandas as pd
from data_loader import DataLoader
from preprocessor import Preprocessor

def run_etl(
    base_path: str,
    output_path: str = None
) -> pd.DataFrame:
    """
    Orchestrate the full ETL workflow:
      1. Load raw CSVs via DataLoader
      2. Merge and preprocess via Preprocessor
      3. Enrich with date features
      4. Optionally save a master DataFrame to disk

    Parameters:
    - base_path: folder containing all source CSV files
    - output_path: optional path (CSV or Parquet) to write the final dataset

    Returns:
    - master_df: the combined DataFrame
    """
    # Ensure base path exists
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")

    # Step 1: Load raw tables
    print("[ETL] Loading data...")
    loader = DataLoader(base_path)
    customers_df     = loader.load_customers()
    orders_df        = loader.load_orders()
    items_df         = loader.load_order_items()
    products_df      = loader.load_products()
    stock_df         = loader.load_stock()
    stock_history_df = loader.load_stock_history()

    print(f"[ETL] Data loaded: customers={customers_df.shape}, orders={orders_df.shape}, "
          f"items={items_df.shape}, products={products_df.shape}")

    # Step 2: Merge key tables
    print("[ETL] Merging data...")
    preprocessor = Preprocessor()
    master_df = preprocessor.merge_all(
        customers_df,
        orders_df,
        items_df,
        products_df
    )
    print(f"[ETL] After merge_all: master_df shape={master_df.shape}")

    # Step 3: Enrich with date-based features
    print("[ETL] Enriching dates...")
    master_df = preprocessor.enrich_dates(master_df, date_col='CreatedOn')
    print(f"[ETL] After enrich_dates: master_df shape={master_df.shape}")

    # TODO: Merge weather forecasts here
    # TODO: Incorporate stock & inventory signals (stock_df, stock_history_df)

    # Step 4: Output final dataset
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        print(f"[ETL] Saving master dataset to {output_path}...")
        if output_path.endswith('.parquet'):
            master_df.to_parquet(output_path, index=False)
        else:
            master_df.to_csv(output_path, index=False)
        print(f"[ETL] Master dataset written to {output_path}")
    else:
        print(f"[ETL] ETL complete. master_df in memory with shape {master_df.shape}.")

    return master_df


if __name__ == '__main__':
    BASE_PATH   = 'data'
    OUTPUT_FILE = 'data/master_dataset.csv'

    # 1. Load raw data
    loader    = DataLoader(BASE_PATH)
    customers = loader.load_customers()
    orders    = loader.load_orders()
    items     = loader.load_order_items()
    products  = loader.load_products()

    # 2. Merge & preprocess
    pre       = Preprocessor()
    master_df = pre.merge_all(customers, orders, items, products)
    master_df = pre.enrich_dates(master_df, date_col='CreatedOn')

    # 3. Save the output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    master_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Master dataset written to {OUTPUT_FILE}")
    print(master_df.head())