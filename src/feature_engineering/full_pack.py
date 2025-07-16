import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import requests
from datetime import datetime
import re

class DataPreparator:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="data_preparation_v1.0")
        self.geocoding_cache = {}
        
    def load_and_inspect_data(self, train_path, test_path, orders_path):
        """Load all datasets and show basic info"""
        print("=" * 60)
        print("LOADING AND INSPECTING DATA")
        print("=" * 60)
        
        # Load datasets
        print(f"Loading {train_path}...")
        train_items = pd.read_csv(train_path, parse_dates=['CreatedOn_x', 'CreatedOn_y'])
        
        print(f"Loading {test_path}...")
        test_items = pd.read_csv(test_path, parse_dates=['CreatedOn_x', 'CreatedOn_y'])
        
        print(f"Loading {orders_path}...")
        orders = pd.read_csv(orders_path, parse_dates=['CreatedOn'])
        
        # Basic info
        print(f"\nDataset shapes:")
        print(f"  Train items: {train_items.shape}")
        print(f"  Test items: {test_items.shape}")
        print(f"  Orders: {orders.shape}")
        
        # Check columns
        print(f"\nTrain items columns: {list(train_items.columns)}")
        print(f"\nOrders columns: {list(orders.columns)}")
        
        # Check for common ID columns
        train_id_cols = [col for col in train_items.columns if 'id' in col.lower() or 'order' in col.lower()]
        orders_id_cols = [col for col in orders.columns if 'id' in col.lower() or 'order' in col.lower()]
        
        print(f"\nTrain ID-related columns: {train_id_cols}")
        print(f"Orders ID-related columns: {orders_id_cols}")
        
        return train_items, test_items, orders
    
    def standardize_column_names(self, train_items, test_items, orders):
        """Standardize column names across datasets"""
        print("\n" + "=" * 60)
        print("STANDARDIZING COLUMN NAMES")
        print("=" * 60)
        
        # Standardize order ID columns
        for df, name in [(train_items, 'train'), (test_items, 'test')]:
            if 'order_id' in df.columns:
                df.rename(columns={'order_id': 'OrdersAfkomstId'}, inplace=True)
                print(f"Renamed 'order_id' to 'OrdersAfkomstId' in {name}")
        
        # Standardize orders columns
        rename_map = {
            'CreatedOn': 'order_date',
            'DeliveryCountry': 'country'
        }
        
        for old_name, new_name in rename_map.items():
            if old_name in orders.columns:
                orders.rename(columns={old_name: new_name}, inplace=True)
                print(f"Renamed '{old_name}' to '{new_name}' in orders")
        
        return train_items, test_items, orders
    
    def analyze_missing_data(self, df, name):
        """Analyze missing data patterns"""
        print(f"\n{name} - Missing Data Analysis:")
        print("-" * 40)
        
        missing_info = []
        for col in df.columns:
            try:
                missing_count = df[col].isnull().sum()
                # Handle the case where sum() returns a Series
                if hasattr(missing_count, 'iloc'):
                    missing_count = missing_count.iloc[0]
                missing_count = int(missing_count)
                missing_pct = (missing_count / len(df)) * 100
                
                if missing_count > 0:
                    missing_info.append({
                        'column': col,
                        'missing_count': missing_count,
                        'missing_pct': round(missing_pct, 2)
                    })
            except Exception as e:
                print(f"Error analyzing column '{col}': {e}")
                continue
        
        if missing_info:
            missing_df = pd.DataFrame(missing_info).sort_values('missing_pct', ascending=False)
            print(missing_df.to_string(index=False))
        else:
            print("No missing data found!")
        
        return missing_info
    
    def geocode_address(self, address_parts):
        """Geocode an address to get coordinates"""
        # Create address string from parts
        address_str = ", ".join([str(part) for part in address_parts if pd.notna(part) and str(part).strip()])
        
        if not address_str or address_str in self.geocoding_cache:
            return self.geocoding_cache.get(address_str, (None, None))
        
        try:
            time.sleep(0.1)  # Rate limiting
            location = self.geolocator.geocode(address_str, timeout=10)
            if location:
                lat, lon = location.latitude, location.longitude
                self.geocoding_cache[address_str] = (lat, lon)
                return lat, lon
            else:
                self.geocoding_cache[address_str] = (None, None)
                return None, None
        except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
            print(f"Geocoding failed for '{address_str}': {e}")
            self.geocoding_cache[address_str] = (None, None)
            return None, None
    
    def get_country_coordinates(self, country):
        """Get default coordinates for a country"""
        country_coords = {
            'Belgium': (50.8503, 4.3517),
            'Netherlands': (52.1326, 5.2913),
            'France': (46.2276, 2.2137),
            'Germany': (51.1657, 10.4515),
            'United Kingdom': (55.3781, -3.4360),
            'Spain': (40.4637, -3.7492),
            'Italy': (41.8719, 12.5674)
        }
        return country_coords.get(country, (50.8503, 4.3517))  # Default to Belgium
    
    def fill_coordinates(self, orders):
        """Fill missing coordinates using geocoding"""
        print("\n" + "=" * 60)
        print("FILLING MISSING COORDINATES")
        print("=" * 60)
        
        # Check current coordinate status
        lat_missing = orders['lat'].isnull().sum()
        lon_missing = orders['lon'].isnull().sum()
        
        print(f"Missing coordinates: lat={lat_missing}, lon={lon_missing}")
        
        if lat_missing == 0 and lon_missing == 0:
            print("All coordinates are already filled!")
            return orders
        
        # Try to geocode based on available address information
        address_columns = [col for col in orders.columns if any(addr_part in col.lower() 
                          for addr_part in ['address', 'city', 'zipcode', 'delivery'])]
        
        print(f"Found address columns: {address_columns}")
        
        # Create missing coordinate mask
        missing_coords = orders['lat'].isnull() | orders['lon'].isnull()
        missing_count = missing_coords.sum()
        
        if missing_count > 0:
            print(f"Attempting to geocode {missing_count} addresses...")
            
            # Try geocoding for missing coordinates
            geocoded_count = 0
            for idx in orders[missing_coords].index:
                if geocoded_count % 100 == 0:
                    print(f"  Processed {geocoded_count}/{missing_count} addresses...")
                
                # Build address from available columns
                address_parts = []
                for col in address_columns:
                    if col in orders.columns:
                        address_parts.append(orders.loc[idx, col])
                
                # Add country if available
                if 'country' in orders.columns:
                    address_parts.append(orders.loc[idx, 'country'])
                
                if address_parts:
                    lat, lon = self.geocode_address(address_parts)
                    if lat and lon:
                        orders.loc[idx, 'lat'] = lat
                        orders.loc[idx, 'lon'] = lon
                        geocoded_count += 1
                
                # Rate limiting
                if geocoded_count % 10 == 0:
                    time.sleep(1)
            
            print(f"Successfully geocoded {geocoded_count} addresses")
        
        # Fill remaining missing coordinates with country defaults
        still_missing = orders['lat'].isnull() | orders['lon'].isnull()
        if still_missing.sum() > 0:
            print(f"Filling {still_missing.sum()} remaining coordinates with country defaults...")
            
            for idx in orders[still_missing].index:
                country = orders.loc[idx, 'country'] if 'country' in orders.columns else 'Belgium'
                lat, lon = self.get_country_coordinates(country)
                orders.loc[idx, 'lat'] = lat
                orders.loc[idx, 'lon'] = lon
        
        # Final check
        final_missing = orders['lat'].isnull().sum() + orders['lon'].isnull().sum()
        print(f"Final missing coordinates: {final_missing}")
        
        return orders
    
    def fill_missing_values(self, df, name):
        """Fill missing values using appropriate strategies"""
        print(f"\n{name} - Filling Missing Values:")
        print("-" * 40)
        
        for col in df.columns:
            try:
                missing_count = df[col].isnull().sum()
                # Handle the case where sum() returns a Series
                if hasattr(missing_count, 'iloc'):
                    missing_count = missing_count.iloc[0]
                missing_count = int(missing_count)
                
                if missing_count == 0:
                    continue
                    
                print(f"Filling {missing_count} missing values in '{col}'...")
                
                if df[col].dtype in ['float64', 'int64']:
                    # Numeric columns - use median
                    fill_value = df[col].median()
                    if pd.notna(fill_value):  # Check if median is not NaN
                        df[col].fillna(fill_value, inplace=True)
                        print(f"  Filled with median: {fill_value}")
                    else:
                        # If all values are NaN, fill with 0
                        df[col].fillna(0, inplace=True)
                        print(f"  Filled with default: 0")
                    
                elif df[col].dtype == 'object':
                    # Text columns - use mode or default
                    if not df[col].dropna().empty:
                        mode_values = df[col].mode()
                        if len(mode_values) > 0:
                            fill_value = mode_values.iloc[0]
                            df[col].fillna(fill_value, inplace=True)
                            print(f"  Filled with mode: {fill_value}")
                        else:
                            df[col].fillna('Unknown', inplace=True)
                            print(f"  Filled with default: 'Unknown'")
                    else:
                        df[col].fillna('Unknown', inplace=True)
                        print(f"  Filled with default: 'Unknown'")
                        
                elif df[col].dtype.name.startswith('datetime'):
                    # Date columns - use median date
                    fill_value = df[col].median()
                    if pd.notna(fill_value):
                        df[col].fillna(fill_value, inplace=True)
                        print(f"  Filled with median date: {fill_value}")
                    else:
                        # If all dates are NaN, use current date
                        current_date = pd.Timestamp.now().normalize()
                        df[col].fillna(current_date, inplace=True)
                        print(f"  Filled with current date: {current_date}")
                        
            except Exception as e:
                print(f"Error filling missing values in column '{col}': {e}")
                continue
        
        return df
    
    def create_bundle_labels(self, df):
        """Create meaningful bundle labels from product data"""
        print("\n" + "=" * 60)
        print("CREATING BUNDLE LABELS")
        print("=" * 60)
        
        # Strategy 1: Use Category if available
        if 'Category' in df.columns:
            df['bundle_label'] = df['Category'].fillna('Uncategorized')
            print("Created bundle labels from 'Category' column")
            print(f"Label distribution:\n{df['bundle_label'].value_counts()}")
            
        # Strategy 2: Use price ranges
        elif 'Price_x' in df.columns:
            df['bundle_label'] = pd.cut(
                df['Price_x'], 
                bins=[0, 20, 50, 100, 200, float('inf')], 
                labels=['Budget', 'Economy', 'Mid-range', 'Premium', 'Luxury'],
                include_lowest=True
            )
            print("Created bundle labels from price ranges")
            print(f"Label distribution:\n{df['bundle_label'].value_counts()}")
            
        # Strategy 3: Use product weight
        elif 'Weight_x' in df.columns:
            df['bundle_label'] = pd.cut(
                df['Weight_x'], 
                bins=[0, 1, 5, 10, float('inf')], 
                labels=['Light', 'Medium', 'Heavy', 'Bulk'],
                include_lowest=True
            )
            print("Created bundle labels from weight ranges")
            print(f"Label distribution:\n{df['bundle_label'].value_counts()}")
            
        # Strategy 4: Default classification
        else:
            df['bundle_label'] = 'Standard'
            print("Created default bundle labels")
        
        # Convert to string to avoid category issues
        df['bundle_label'] = df['bundle_label'].astype(str)
        
        return df
    
    def prepare_final_datasets(self, train_items, test_items, orders):
        """Merge and prepare final datasets"""
        print("\n" + "=" * 60)
        print("PREPARING FINAL DATASETS")
        print("=" * 60)
        
        # Select relevant columns from orders
        orders_clean = orders[['OrdersAfkomstId', 'order_date', 'lat', 'lon', 'country']].copy()
        
        # Merge items with orders
        print("Merging train items with orders...")
        train_final = train_items.merge(orders_clean, on='OrdersAfkomstId', how='left')
        
        print("Merging test items with orders...")
        test_final = test_items.merge(orders_clean, on='OrdersAfkomstId', how='left')
        
        # Add country codes for holidays
        country_map = {
            'Belgium': 'BE',
            'Netherlands': 'NL', 
            'France': 'FR',
            'Germany': 'DE',
            'United Kingdom': 'GB',
            'Spain': 'ES',
            'Italy': 'IT'
        }
        
        train_final['country_code'] = train_final['country'].map(country_map)
        test_final['country_code'] = test_final['country'].map(country_map)
        
        # Normalize dates
        train_final['order_date'] = pd.to_datetime(train_final['order_date']).dt.normalize()
        test_final['order_date'] = pd.to_datetime(test_final['order_date']).dt.normalize()
        
        # Create bundle labels
        train_final = self.create_bundle_labels(train_final)
        test_final = self.create_bundle_labels(test_final)
        
        print(f"\nFinal dataset shapes:")
        print(f"  Train: {train_final.shape}")
        print(f"  Test: {test_final.shape}")
        
        return train_final, test_final
    
    def save_cleaned_data(self, train_final, test_final, orders, output_dir='data/cleaned/'):
        """Save cleaned datasets"""
        print("\n" + "=" * 60)
        print("SAVING CLEANED DATA")
        print("=" * 60)
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save datasets
        train_path = f"{output_dir}train_data_cleaned.csv"
        test_path = f"{output_dir}test_data_cleaned.csv"
        orders_path = f"{output_dir}orders_cleaned.csv"
        
        train_final.to_csv(train_path, index=False)
        test_final.to_csv(test_path, index=False)
        orders.to_csv(orders_path, index=False)
        
        print(f"Saved cleaned datasets:")
        print(f"  Train: {train_path}")
        print(f"  Test: {test_path}")
        print(f"  Orders: {orders_path}")
        
        return train_path, test_path, orders_path

def main():
    """Main data preparation pipeline"""
    # File paths
    TRAIN_PATH = 'data/work_material/train_data.csv'
    TEST_PATH = 'data/work_material/test_data.csv'
    ORDERS_PATH = 'data/weather_api/orders_with_coords_final.csv'
    
    # Initialize preparator
    preparator = DataPreparator()
    
    try:
        # Step 1: Load and inspect
        train_items, test_items, orders = preparator.load_and_inspect_data(
            TRAIN_PATH, TEST_PATH, ORDERS_PATH
        )
        
        # Step 2: Standardize column names
        train_items, test_items, orders = preparator.standardize_column_names(
            train_items, test_items, orders
        )
        
        # Step 3: Analyze missing data
        preparator.analyze_missing_data(train_items, "TRAIN ITEMS")
        preparator.analyze_missing_data(test_items, "TEST ITEMS")
        preparator.analyze_missing_data(orders, "ORDERS")
        
        # Step 4: Fill coordinates
        orders = preparator.fill_coordinates(orders)
        
        # Step 5: Fill missing values
        train_items = preparator.fill_missing_values(train_items, "TRAIN ITEMS")
        test_items = preparator.fill_missing_values(test_items, "TEST ITEMS")
        orders = preparator.fill_missing_values(orders, "ORDERS")
        
        # Step 6: Prepare final datasets
        train_final, test_final = preparator.prepare_final_datasets(
            train_items, test_items, orders
        )
        
        # Step 7: Save cleaned data
        preparator.save_cleaned_data(train_final, test_final, orders)
        
        print("\n" + "=" * 60)
        print("DATA PREPARATION COMPLETE!")
        print("=" * 60)
        print("You can now use the cleaned datasets for machine learning.")
        
    except Exception as e:
        print(f"\nError during data preparation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()