import pandas as pd

class Preprocessor:
    """
    Class for merging and preprocessing data tables into a single master DataFrame.
    """

    def merge_customers_orders(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge orders with customer lifetime metrics.

        Parameters:
        - orders_df: DataFrame containing orders
        - customers_df: DataFrame containing customer info

        Returns:
        - Merged DataFrame with customer lifetime orders and value
        """
        merged = orders_df.merge(
            customers_df[['Id', 'LifetimeOrders', 'LifetimeValue']],
            left_on='IdCustomer',
            right_on='Id',
            how='left'
        )
        # Rename columns for clarity
        merged = merged.rename(columns={
            'LifetimeOrders': 'cust_lifetime_orders',
            'LifetimeValue': 'cust_lifetime_value'
        })
        return merged

    def merge_items_products(
        self,
        items_df: pd.DataFrame,
        products_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge order items with product information: name, price, current stock.

        Parameters:
        - items_df: DataFrame of order items
        - products_df: DataFrame of product details

        Returns:
        - Merged DataFrame with product metadata
        """
        merged = items_df.merge(
            products_df[['Id', 'Name', 'Price', 'TotalStock']],
            left_on='ProductId',
            right_on='Id',
            how='left'
        )
        # Rename for convenience
        merged = merged.rename(columns={
            'Name': 'product_name',
            'Price': 'unit_price',
            'TotalStock': 'current_stock'
        })
        return merged

    def aggregate_order_items(
        self,
        items_enriched: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate item-level data into order-level features.

        Parameters:
        - items_enriched: DataFrame returned by merge_items_products

        Returns:
        - DataFrame with one row per order and aggregated metrics
        """
        # Group by order and compute aggregates
        agg = (
            items_enriched
            .groupby('OrdersId')
            .agg(
                total_items=('Amount', 'sum'),
                avg_item_price=('item_price', 'mean'),
                total_stock_value=('Amount', lambda x: (items_enriched.loc[x.index, 'current_stock'] * x).sum())
            )
            .reset_index()
        )
        return agg

    def merge_all(
        self,
        customers_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        items_df: pd.DataFrame,
        products_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Complete merge pipeline combining customers, orders, items, and products.

        Parameters:
        - customers_df: customer DataFrame
        - orders_df: orders DataFrame
        - items_df: order items DataFrame
        - products_df: products DataFrame

        Returns:
        - Master DataFrame ready for feature engineering
        """
        # Step 1: Merge orders with customers
        df = self.merge_customers_orders(orders_df, customers_df)

        

        # Step 2: Enrich items with product info
        items_enriched = self.merge_items_products(items_df, products_df)

        # DEBUG prints
        print("=== Debug: items_enriched columns ===")
        print(items_enriched.columns.tolist())
        print("=== Debug: items_enriched head ===")
        print(items_enriched.head())

        # Step 3: Aggregate items to order level
        order_agg = self.aggregate_order_items(items_enriched)

        # Step 4: Merge aggregated metrics back into orders
        master_df = df.merge(
            order_agg,
            left_on='Id',
            right_on='OrdersId',
            how='left'
        )
        return master_df

    def enrich_dates(
        self,
        df: pd.DataFrame,
        date_col: str = 'CreatedOn'
    ) -> pd.DataFrame:
        """
        Extract date-based features: month, day of week, weekend flag, season.

        Parameters:
        - df: DataFrame with a datetime column
        - date_col: name of the datetime column to parse

        Returns:
        - DataFrame with new date feature columns
        """
        # Ensure datetime type
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Extract basic date parts
        df['month'] = df[date_col].dt.month
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

        # Map month to season
        def month_to_season(m):
            if 3 <= m <= 5:
                return 'spring'
            if 6 <= m <= 8:
                return 'summer'
            if 9 <= m <= 11:
                return 'autumn'
            return 'winter'

        df['season'] = df['month'].apply(month_to_season)
        return df
