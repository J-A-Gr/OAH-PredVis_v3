import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_dataframe(df: pd.DataFrame, name: str = "Dataset", figsize: tuple = (15, 10)):
    """
    Perform comprehensive exploratory analysis on a DataFrame with visualizations.
    
    Args:
        df: pandas DataFrame to analyze
        name: Name of the dataset for reporting
        figsize: Figure size for plots (width, height)
    
    Prints and visualizes:
    - Basic information (shape, memory usage)
    - Data types distribution
    - Missing values analysis with heatmap
    - Duplicate rows analysis
    - Outliers detection for numeric columns
    - Distribution plots for numeric columns
    - Value counts for categorical columns
    - Correlation matrix for numeric columns
    """
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE ANALYSIS: {name}")
    print(f"{'='*60}\n")
    
    # 1. BASIC INFORMATION
    print("1. BASIC INFORMATION")
    print("-" * 40)
    print(f"Shape: {df.shape} (rows, columns)")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Dataset Range: {len(df)} rows\n")
    
    # Data types summary
    dtype_counts = df.dtypes.value_counts()
    print("Data Types Distribution:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    print()
    
    # 2. MISSING VALUES ANALYSIS
    print("\n2. MISSING VALUES ANALYSIS")
    print("-" * 40)
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print("Columns with Missing Values:")
        print(missing_df.to_string(index=False))
        
        # Missing values heatmap
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title(f'Missing Values Heatmap - {name}')
        plt.tight_layout()
        
        # Missing values bar plot
        plt.subplot(1, 2, 2)
        missing_df.plot(x='Column', y='Missing_Percentage', kind='bar', legend=False, ax=plt.gca())
        plt.title('Missing Values Percentage by Column')
        plt.xlabel('Columns')
        plt.ylabel('Missing %')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No missing values found!")
    
    # 3. DUPLICATE ROWS ANALYSIS
    print("\n\n3. DUPLICATE ROWS ANALYSIS")
    print("-" * 40)
    duplicates = df.duplicated().sum()
    print(f"Total Duplicate Rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    if duplicates > 0:
        print("\nFirst 5 duplicate rows:")
        print(df[df.duplicated(keep='first')].head())
    
    # 4. NUMERIC COLUMNS ANALYSIS
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        print("\n\n4. NUMERIC COLUMNS ANALYSIS")
        print("-" * 40)
        print(f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}", 
              "..." if len(numeric_cols) > 5 else "")
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(df[numeric_cols].describe().round(2))
        
        # Outliers Detection
        print("\nOutliers Detection (using IQR method):")
        outliers_summary = []
        
        for col in numeric_cols:
            # Skip if column has no variance
            if df[col].std() == 0 or df[col].nunique() <= 1:
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Skip if IQR is 0 (all values in the middle 50% are the same)
            if IQR == 0:
                continue
                
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outliers_summary.append({
                    'Column': col,
                    'Outlier_Count': outlier_count,
                    'Outlier_%': round(outlier_count/len(df)*100, 2),
                    'Lower_Bound': round(lower_bound, 2),
                    'Upper_Bound': round(upper_bound, 2)
                })
        
        if outliers_summary:
            outliers_df = pd.DataFrame(outliers_summary)
            print(outliers_df.to_string(index=False))
            
            # Outlier visualization
            # 1. Bar plot of outlier percentages
            plt.figure(figsize=(10, 6))
            outliers_df = outliers_df.sort_values('Outlier_%', ascending=False)
            colors = ['red' if x > 10 else 'orange' if x > 5 else 'yellow' for x in outliers_df['Outlier_%']]
            bars = plt.bar(outliers_df['Column'], outliers_df['Outlier_%'], color=colors)
            plt.title(f'Outlier Percentage by Column - {name}', fontsize=14)
            plt.xlabel('Columns')
            plt.ylabel('Outlier Percentage (%)')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            plt.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
            plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10% threshold')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        # Z-score based outlier detection
        print("\n\nOutliers Detection (using Z-score method, threshold=3):")
        zscore_outliers = []
        
        for col in numeric_cols:
            if df[col].std() == 0 or df[col].nunique() <= 1:
                continue
            
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_count = (z_scores > 3).sum()
            
            if outlier_count > 0:
                zscore_outliers.append({
                    'Column': col,
                    'Outlier_Count': outlier_count,
                    'Outlier_%': round(outlier_count/len(df[col].dropna())*100, 2)
                })
        
        if zscore_outliers:
            zscore_df = pd.DataFrame(zscore_outliers)
            print("Columns with extreme outliers (|Z-score| > 3):")
            print(zscore_df.to_string(index=False))
            
            # Violin plots for outlier visualization
            cols_with_outliers = [item['Column'] for item in outliers_summary]
            if cols_with_outliers:
                n_cols_violin = min(4, len(cols_with_outliers))
                n_rows_violin = (len(cols_with_outliers) + n_cols_violin - 1) // n_cols_violin
                
                fig, axes = plt.subplots(n_rows_violin, n_cols_violin, figsize=(15, 4*n_rows_violin))
                if len(cols_with_outliers) == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten() if len(cols_with_outliers) > 1 else [axes]
                
                for idx, col in enumerate(cols_with_outliers[:len(axes)]):
                    ax = axes[idx]
                    
                    # Create violin plot
                    parts = ax.violinplot([df[col].dropna()], positions=[1], showmeans=True, 
                                         showextrema=True, showmedians=True)
                    
                    # Color the violin
                    for pc in parts['bodies']:
                        pc.set_facecolor('lightblue')
                        pc.set_alpha(0.7)
                    
                    # Overlay scatter plot of actual points
                    y = df[col].dropna()
                    x = np.random.normal(1, 0.04, size=len(y))
                    
                    # Color outliers differently
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outlier_mask = (y < lower_bound) | (y > upper_bound)
                        
                        # Plot normal points
                        ax.scatter(x[~outlier_mask], y[~outlier_mask], alpha=0.3, s=10, color='blue')
                        # Plot outliers
                        ax.scatter(x[outlier_mask], y[outlier_mask], alpha=0.8, s=30, color='red', 
                                 marker='D', label='Outliers')
                    else:
                        ax.scatter(x, y, alpha=0.3, s=10, color='blue')
                    
                    ax.set_title(f'Violin Plot: {col}')
                    ax.set_ylabel(col)
                    ax.set_xticks([1])
                    ax.set_xticklabels([col])
                    ax.grid(True, alpha=0.3)
                
                # Hide empty subplots
                for idx in range(len(cols_with_outliers), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.suptitle(f'Violin Plots with Outliers - {name}\n' + 
                           'Blue dots: Normal values, Red diamonds: Outliers', fontsize=16)
                plt.tight_layout()
                plt.show()
            
            # 2. Scatter plot matrix for outliers (if we have 2-4 numeric columns)
            valid_cols_for_scatter = [col for col in numeric_cols 
                                     if df[col].std() > 0 and col in outliers_df['Column'].values]
            
            if 2 <= len(valid_cols_for_scatter) <= 4:
                from pandas.plotting import scatter_matrix
                
                # Create a column to mark outliers
                outlier_mask = pd.Series(False, index=df.index)
                for col in valid_cols_for_scatter:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outlier_mask |= (df[col] < lower_bound) | (df[col] > upper_bound)
                
                # Create scatter matrix with outliers highlighted
                fig, axes = plt.subplots(len(valid_cols_for_scatter), len(valid_cols_for_scatter), 
                                       figsize=(12, 12))
                
                # Plot normal points
                scatter_matrix(df[valid_cols_for_scatter][~outlier_mask], 
                             alpha=0.5, figsize=(12, 12), diagonal='hist',
                             ax=axes, color='blue', hist_kwds={'bins': 30})
                
                # Overlay outliers in red
                if outlier_mask.sum() > 0:
                    scatter_matrix(df[valid_cols_for_scatter][outlier_mask], 
                                 alpha=0.8, figsize=(12, 12), diagonal='hist',
                                 ax=axes, color='red', hist_kwds={'bins': 30})
                
                plt.suptitle(f'Scatter Matrix with Outliers Highlighted - {name}\n' + 
                           f'Blue: Normal points, Red: Outliers', fontsize=16)
                plt.tight_layout()
                plt.show()
        else:
            print("No outliers detected using IQR method.")
        
        # Distribution plots for numeric columns
        n_numeric = len(numeric_cols)
        if n_numeric > 0:
            n_cols = min(3, n_numeric)
            n_rows = (n_numeric + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten() if n_numeric > 1 else [axes]
            
            for idx, col in enumerate(numeric_cols):
                if idx < len(axes):
                    ax = axes[idx]
                    
                    # Histogram
                    df[col].hist(bins=30, ax=ax, alpha=0.7, edgecolor='black')
                    
                    # Try to add KDE if possible
                    try:
                        # Check if column has enough unique values for KDE
                        if df[col].nunique() > 1 and len(df[col].dropna()) > 1:
                            ax2 = ax.twinx()
                            df[col].plot(kind='kde', ax=ax2, color='red', linewidth=2)
                            ax2.set_ylabel('')
                            ax2.set_yticks([])
                    except Exception as e:
                        # If KDE fails, just continue without it
                        pass
                    
                    ax.set_title(f'Distribution of {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    
                    # Add mean and median lines
                    mean_val = df[col].mean()
                    median_val = df[col].median()
                    ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
                    ax.legend()
            
            # Hide empty subplots
            for idx in range(n_numeric, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Numeric Columns Distributions - {name}', fontsize=16)
            plt.tight_layout()
            plt.show()
        
        # Box plots for outlier visualization
        if n_numeric > 0:
            n_cols = min(3, n_numeric)
            n_rows = (n_numeric + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten() if n_numeric > 1 else [axes]
            
            for idx, col in enumerate(numeric_cols):
                if idx < len(axes):
                    ax = axes[idx]
                    
                    # Create box plot with customization
                    box_plot = df.boxplot(column=col, ax=ax, patch_artist=True,
                                         showmeans=True, meanline=True,
                                         medianprops={'color': 'red', 'linewidth': 2},
                                         meanprops={'color': 'green', 'linewidth': 2},
                                         flierprops={'marker': 'o', 'markerfacecolor': 'red', 
                                                   'markersize': 8, 'alpha': 0.5})
                    
                    ax.set_title(f'Box Plot of {col}')
                    ax.set_ylabel(col)
                    
                    # Add outlier count annotation
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
                        
                        if outlier_count > 0:
                            ax.text(0.02, 0.98, f'Outliers: {outlier_count}', 
                                   transform=ax.transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            
            # Hide empty subplots
            for idx in range(n_numeric, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Box Plots for Outlier Detection - {name}\n' + 
                        'Red line: Median, Green line: Mean, Red dots: Outliers', fontsize=16)
            plt.tight_layout()
            plt.show()
        
        # Correlation Matrix
        if len(numeric_cols) > 1:
            print("\n\nCorrelation Analysis:")
            # Filter out columns with no variance
            valid_numeric_cols = [col for col in numeric_cols if df[col].std() > 0]
            
            if len(valid_numeric_cols) > 1:
                corr_matrix = df[valid_numeric_cols].corr()
                
                plt.figure(figsize=(10, 8))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                           cmap='coolwarm', vmin=-1, vmax=1, center=0,
                           square=True, linewidths=0.5)
                plt.title(f'Correlation Matrix - {name}', fontsize=16)
                plt.tight_layout()
                plt.show()
            else:
                print("Not enough columns with variance for correlation analysis.")
    
    # 5. CATEGORICAL COLUMNS ANALYSIS
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        print("\n\n5. CATEGORICAL COLUMNS ANALYSIS")
        print("-" * 40)
        print(f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}", 
              "..." if len(categorical_cols) > 5 else "")
        
        # Value counts for categorical columns
        for col in categorical_cols[:5]:  # Limit to first 5 to avoid too many plots
            print(f"\n{col} - Unique values: {df[col].nunique()}")
            value_counts = df[col].value_counts().head(10)
            print(value_counts)
            
            if df[col].nunique() <= 20:  # Only plot if not too many categories
                plt.figure(figsize=(10, 6))
                
                # Bar plot
                plt.subplot(1, 2, 1)
                value_counts.plot(kind='bar')
                plt.title(f'Top Values in {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                
                # Pie chart
                plt.subplot(1, 2, 2)
                value_counts.head(10).plot(kind='pie', autopct='%1.1f%%')
                plt.title(f'Distribution of {col} (Top 10)')
                plt.ylabel('')
                
                plt.tight_layout()
                plt.show()
    
    # 6. DATETIME COLUMNS ANALYSIS
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if datetime_cols:
        print("\n\n6. DATETIME COLUMNS ANALYSIS")
        print("-" * 40)
        print(f"Datetime columns ({len(datetime_cols)}): {', '.join(datetime_cols)}")
        
        for col in datetime_cols:
            print(f"\n{col}:")
            print(f"  Date range: {df[col].min()} to {df[col].max()}")
            print(f"  Missing values: {df[col].isnull().sum()}")
            
            # Time series plot
            if df[col].notna().sum() > 0:
                plt.figure(figsize=(12, 4))
                df[col].value_counts().sort_index().plot()
                plt.title(f'Time Series Distribution - {col}')
                plt.xlabel('Date')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
    
    # 7. SUMMARY RECOMMENDATIONS
    print("\n\n7. SUMMARY & RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = []
    
    if len(missing_df) > 0:
        high_missing = missing_df[missing_df['Missing_Percentage'] > 50]
        if len(high_missing) > 0:
            recommendations.append(f"- Consider dropping columns with >50% missing values: {', '.join(high_missing['Column'].tolist())}")
        else:
            recommendations.append("- Consider imputation strategies for columns with missing values")
    
    if duplicates > 0:
        recommendations.append(f"- Remove {duplicates} duplicate rows")
    
    if outliers_summary:
        recommendations.append("- Investigate and handle outliers in numeric columns")
    
    if len(categorical_cols) > 0:
        high_cardinality = [col for col in categorical_cols if df[col].nunique() > 50]
        if high_cardinality:
            recommendations.append(f"- High cardinality categorical columns may need encoding: {', '.join(high_cardinality[:3])}")
    
    if recommendations:
        print("Recommendations:")
        for rec in recommendations:
            print(rec)
    else:
        print("Dataset appears to be clean and ready for analysis!")
    
    return {
        'missing_summary': missing_df,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'datetime_cols': datetime_cols,
        'outliers_summary': outliers_summary if 'outliers_summary' in locals() else []
    }


# Example usage:
if __name__ == "__main__":
    # Set visualization style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Analyze orders
    # orders_df = pd.read_csv("data/cleaned/orders_clean.csv", parse_dates=["CreatedOn"])
    # analysis_results = analyze_dataframe(orders_df, "Orders")
    
    # You can also analyze other datasets:
    # customers_df = pd.read_csv("data\cleaned\customers_clean.csv")
    # analyze_dataframe(customers_df, "Customers")
    
    items_df = pd.read_csv("data\cleaned\order_items_clean.csv", parse_dates=["CreatedOn"])
    analyze_dataframe(items_df, "OrderItems")
    
    # products_df = pd.read_csv("data/oahdotnet_live_Products.csv")
    # analyze_dataframe(products_df, "Products")
    
    # stock_df = pd.read_csv("data/oahdotnet_live_ProductsStock.csv")
    # analyze_dataframe(stock_df, "Stock")
    
    # stock_hist_df = pd.read_csv("data/oahdotnet_live_StockHistory.csv", parse_dates=["UpdatedOn"])
    # analyze_dataframe(stock_hist_df, "StockHistory")