import os
import pandas as pd
import numpy as np
import time

# Dask imports
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration for Dataset Generation and Dask ---
OUTPUT_CSV_FILE = "large_transaction_data.csv"
NUM_ROWS = 5_000_000 # Number of rows for the dummy dataset
DASK_N_WORKERS = 4   # Number of Dask workers (processes)
DASK_THREADS_PER_WORKER = 1 # Set to 1 for multiprocessing on Windows for better stability
DASK_MEMORY_LIMIT = '4GB' # Memory limit per worker. Adjust based on your total RAM.

# --- Data Generation Function ---
def generate_large_csv(filename, num_rows):
    """Generates a large CSV file with dummy transaction data."""
    print(f"Generating {num_rows:,} rows of dummy data for '{filename}'...")
    start_time = time.time()

    products = ["Laptop", "Monitor", "Keyboard", "Mouse", "Webcam", "Headphones", "Microphone", "Speaker"]
    regions = ["North", "South", "East", "West", "Central"]
    customers = [f"Customer_{i}" for i in range(500)]

    chunk_size = 100_000
    if num_rows < chunk_size:
        chunk_size = num_rows

    with open(filename, 'w') as f:
        # Write header
        header = "TransactionID,CustomerID,ProductID,Region,Quantity,UnitPrice,TransactionTimestamp"
        f.write(header + "\n")

        for i in range(0, num_rows, chunk_size):
            current_num_rows = min(chunk_size, num_rows - i)
            data = {
                "TransactionID": np.arange(i + 1, i + current_num_rows + 1),
                "CustomerID": np.random.choice(customers, current_num_rows),
                "ProductID": np.random.choice(products, current_num_rows),
                "Region": np.random.choice(regions, current_num_rows),
                "Quantity": np.random.randint(1, 10, current_num_rows),
                "UnitPrice": np.round(np.random.uniform(10.0, 500.0, current_num_rows), 2),
                "TransactionTimestamp": pd.to_datetime(pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365*24*60*60, current_num_rows), unit='s'))
            }
            chunk_df = pd.DataFrame(data)
            chunk_df.to_csv(f, header=False, index=False, mode='a')
            if (i // chunk_size) % 10 == 0:
                print(f"  Generated {i + current_num_rows:,} rows...", end='\r')
    end_time = time.time()
    print(f"\nFinished generating data in {end_time - start_time:.2f} seconds. File size: {os.path.getsize(filename)/(1024*1024):.2f} MB")

### Main Execution Block

if __name__ == '__main__':
    cluster = None # Initialize cluster to None for finally block
    client = None  # Initialize client to None for finally block

    try:
        # --- Initialize Dask Client ---
        print("\n--- Initializing Dask Client ---")
        # LocalCluster uses multiprocessing by default.
        # Explicitly setting processes=True and threads_per_worker=1 can enhance stability on Windows.
        cluster = LocalCluster(n_workers=DASK_N_WORKERS,
                               threads_per_worker=DASK_THREADS_PER_WORKER,
                               memory_limit=DASK_MEMORY_LIMIT,
                               processes=True) # Ensure workers are separate processes
        client = Client(cluster)
        print(f"Dask Client created. Dashboard link: {client.dashboard_link}")
        print(f"Using {len(client.scheduler_info()['workers'])} workers.")

        # --- Load the Dataset into a Dask DataFrame ---
        # Generate large data if the file doesn't exist
        if not os.path.exists(OUTPUT_CSV_FILE):
            generate_large_csv(OUTPUT_CSV_FILE, NUM_ROWS)
        else:
            print(f"'{OUTPUT_CSV_FILE}' already exists. Skipping data generation.")

        print(f"\n--- Loading '{OUTPUT_CSV_FILE}' into Dask DataFrame ---")
        try:
            # Dask's read_csv is robust; providing 'parse_dates' is sufficient for date columns.
            # Avoid using 'dtype' for columns also specified in 'parse_dates' to prevent conflicts.
            dask_df = dd.read_csv(OUTPUT_CSV_FILE,
                                  parse_dates=['TransactionTimestamp'],
                                  blocksize='64MB') # Read file in 64MB blocks for partitioning

            print("Dataset loaded successfully.")
            print("\nDask DataFrame partitions:")
            print(dask_df.npartitions)
            print("\nDataFrame Schema:")
            print(dask_df.dtypes)
            print("\nFirst 5 rows (computed):")
            print(dask_df.head(5)) # .head() triggers computation of first few rows
            print(f"Total rows in DataFrame (computed): {len(dask_df):,}") # len() triggers computation

        except Exception as e:
            print(f"Error loading data into Dask DataFrame: {e}")
            print("Please ensure the CSV file exists and its content matches expected format.")
            # Re-raise the exception after printing, or exit, as the Dask operations can't proceed
            raise # Re-raise to fall into the outer except block for cleanup
            # exit() # Alternatively, you could exit here

        # --- Exploratory Data Analysis (EDA) with Dask ---
        print("\n--- Performing Basic EDA with Dask ---")

        print("\nMissing Values Count (computed):")
        missing_counts = dask_df.isnull().sum().compute()
        print(missing_counts)

        print("\nSummary Statistics for Numerical Columns (computed):")
        print(dask_df[['Quantity', 'UnitPrice']].describe().compute())

        dask_df['TotalPrice'] = dask_df['Quantity'] * dask_df['UnitPrice']
        print("\nDataFrame with 'TotalPrice' column added (first 5 rows, computed):")
        print(dask_df.head(5))

        print("\nTop 10 Products by Total Sales (computed):")
        top_products = dask_df.groupby("ProductID")['TotalPrice'].sum().nlargest(10).compute()
        print(top_products)

        print("\nTotal Sales by Region (computed):")
        sales_by_region = dask_df.groupby("Region")['TotalPrice'].sum().compute()
        print(sales_by_region)

        print("\nDaily Sales Trend (first 30 days for plotting, computed):")
        dask_df['TransactionDate'] = dask_df['TransactionTimestamp'].dt.date
        daily_sales = dask_df.groupby("TransactionDate")['TotalPrice'].sum()
        daily_sales_pd = daily_sales.compute().head(30)

        plt.figure(figsize=(12, 6))
        daily_sales_pd.index = pd.to_datetime(daily_sales_pd.index)
        sns.lineplot(data=daily_sales_pd, x=daily_sales_pd.index, y=daily_sales_pd.values)
        plt.title("Daily Sales Trend (First 30 Days of Transactions)")
        plt.xlabel("Date")
        plt.ylabel("Daily Sales")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("\n--- EDA complete ---")

        # --- Demonstrating Scalability & Performance Insights ---
        print("\n--- Demonstrating Scalability with a Complex Aggregation ---")

        print("Calculating Average Quantity Sold per Product, Region, and Month...")
        start_time = time.time()

        monthly_product_region_avg = dask_df.assign(TransactionMonth=dask_df['TransactionTimestamp'].dt.month) \
                                              .groupby(["ProductID", "Region", "TransactionMonth"])['Quantity'].mean()

        result_df = monthly_product_region_avg.compute().reset_index()
        result_df = result_df.sort_values(by='Quantity', ascending=False)

        print("First 10 rows of Average Quantity:")
        print(result_df.head(10))

        end_time = time.time()

        print(f"Time taken for complex aggregation on {NUM_ROWS:,} rows: {end_time - start_time:.2f} seconds")

        print("\n--- Insights on Scalability with Dask ---")
        print(f"For a dataset of {NUM_ROWS:,} rows (approx. {os.path.getsize(OUTPUT_CSV_FILE)/(1024*1024):.2f} MB),")
        print(f"Dask successfully processed the complex aggregation by parallelizing tasks across {DASK_N_WORKERS} workers,")
        print("avoiding out-of-memory errors that would occur with Pandas for much larger datasets.")
        print("\nKey aspects demonstrating Dask's scalability:")
        print("1.  **Pure Python:** Dask integrates seamlessly with existing Python libraries like Pandas and NumPy.")
        print("2.  **Task Graph:** Dask builds a task graph, allowing for lazy evaluation and optimization before execution.")
        print("3.  **Parallel Execution:** Computations are divided into smaller chunks and executed in parallel across threads or processes (local) or distributed across a cluster.")
        print("4.  **Memory Management:** Dask intelligently handles data that doesn't fit into memory by spilling to disk when necessary.")
        print("5.  **Dashboard:** The Dask dashboard (link printed above) provides real-time visualization of computation, memory usage, and task scheduling, offering insight into parallelism.")
        print("Dask can scale from a single machine to large clusters, making it a powerful tool for Big Data analysis in Python.")

        # Example: Writing the processed data to Parquet (a common Big Data format)
        OUTPUT_PARQUET_DIR = "processed_transaction_data_dask.parquet"
        print(f"\n--- Saving Processed Data to Parquet: '{OUTPUT_PARQUET_DIR}' ---")
        start_time = time.time()

        # Dask will create a directory with multiple parquet files inside
        dask_df.to_parquet(OUTPUT_PARQUET_DIR, write_metadata_file=True, overwrite=True)

        end_time = time.time()
        print(f"Time taken to write {len(dask_df):,} rows to Parquet: {end_time - start_time:.2f} seconds")
        print(f"Check the directory '{OUTPUT_PARQUET_DIR}' to see the partitioned output files.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for detailed error info
    finally:
        # --- Close Dask Client and Cluster (Crucial for cleanup) ---
        # Ensure client and cluster objects exist before trying to close them
        if client:
            client.close()
            print("\nDask Client closed.")
        if cluster:
            cluster.close()
            print("Dask LocalCluster closed.")
        print("\nProgram finished (Dask resources cleaned up).")
