import pandas as pd

class DataLoader:
    def __init__(self, base_path):
        self.base = base_path

    def load_customers(self):
        df = pd.read_csv(f"{self.base}/oahdotnet_live_Customers.csv")
        df['UpdatedOn'] = pd.to_datetime(df['UpdatedOn'], errors='coerce')
        return df.drop_duplicates()

    def load_orders(self):
        df = pd.read_csv(f"{self.base}/oahdotnet_live_Orders.csv")
        df['CreatedOn'] = pd.to_datetime(df['CreatedOn'], errors='coerce')
        return df.drop_duplicates()

    def load_order_items(self):
        df = pd.read_csv(f"{self.base}/oahdotnet_live_OrdersItem.csv")
        df['CreatedOn'] = pd.to_datetime(df['CreatedOn'], errors='coerce')
        return df.drop_duplicates()

    def load_products(self):
        df = pd.read_csv(f"{self.base}/oahdotnet_live_Products.csv")
        # parse any date fields if needed
        return df.drop_duplicates()

    def load_stock(self):
        df = pd.read_csv(f"{self.base}/oahdotnet_live_ProductsStock.csv")
        # no dates here, but check numeric types
        return df.drop_duplicates()

    def load_stock_history(self):
        df = pd.read_csv(f"{self.base}/oahdotnet_live_StockHistory.csv")
        df['UpdatedOn'] = pd.to_datetime(df['UpdatedOn'], errors='coerce')
        return df.drop_duplicates()

