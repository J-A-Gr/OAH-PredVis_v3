import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedBundleRecommendationSystem:
    """
    Advanced ML system for product bundle recommendations with support for
    product-level data and collaborative filtering
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.customer_model = None
        self.product_model = None
        self.bundle_optimizer = None
        self.recommendation_cache = {}
        
    def load_order_data(self, orders_filepath, products_filepath=None):
        """Load order data and optionally product-level data"""
        self.orders_df = pd.read_csv(orders_filepath)
        self.orders_df['CreatedOn'] = pd.to_datetime(self.orders_df['CreatedOn'])
        
        # Filter completed orders
        self.orders_completed = self.orders_df[self.orders_df['Status'] == 'completed'].copy()
        
        if products_filepath:
            self.products_df = pd.read_csv(products_filepath)
            self.has_product_data = True
        else:
            self.has_product_data = False
            
        print(f"Loaded {len(self.orders_completed)} completed orders")
        if self.has_product_data:
            print(f"Loaded product-level data")
            
    def create_rfm_features(self):
        """Create RFM (Recency, Frequency, Monetary) features for customers"""
        current_date = self.orders_completed['CreatedOn'].max()
        
        rfm = self.orders_completed.groupby('IdCustomer').agg({
            'CreatedOn': lambda x: (current_date - x.max()).days,  # Recency
            'Id': 'count',  # Frequency
            'TotalPrice': 'sum'  # Monetary
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Create RFM scores
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        # Combine scores
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        return rfm
    
    def create_customer_embedding(self):
        """Create rich customer embeddings for recommendation"""
        customer_features = []
        
        for customer_id in self.orders_completed['IdCustomer'].unique():
            customer_orders = self.orders_completed[self.orders_completed['IdCustomer'] == customer_id]
            
            # Time-based features
            order_dates = customer_orders['CreatedOn'].sort_values()
            if len(order_dates) > 1:
                inter_purchase_times = [(order_dates.iloc[i+1] - order_dates.iloc[i]).days 
                                       for i in range(len(order_dates)-1)]
                avg_purchase_interval = np.mean(inter_purchase_times)
                purchase_regularity = np.std(inter_purchase_times) if len(inter_purchase_times) > 1 else 0
            else:
                avg_purchase_interval = 0
                purchase_regularity = 0
            
            # Seasonal patterns
            order_months = customer_orders['CreatedOn'].dt.month
            seasonal_concentration = order_months.value_counts().max() / len(order_months)
            
            # Price sensitivity
            prices = customer_orders['TotalPrice']
            price_variance = prices.var()
            
            # Geographic features
            unique_cities = customer_orders['DeliveryCity'].nunique()
            
            features = {
                'IdCustomer': customer_id,
                'total_orders': len(customer_orders),
                'total_revenue': customer_orders['TotalPrice'].sum(),
                'avg_order_value': customer_orders['TotalPrice'].mean(),
                'price_variance': price_variance,
                'avg_products_per_order': customer_orders['TotalProducts'].mean(),
                'max_products_in_order': customer_orders['TotalProducts'].max(),
                'avg_purchase_interval': avg_purchase_interval,
                'purchase_regularity': purchase_regularity,
                'seasonal_concentration': seasonal_concentration,
                'preferred_month': order_months.mode()[0] if len(order_months.mode()) > 0 else 0,
                'unique_delivery_cities': unique_cities,
                'shipping_preference': customer_orders['ShippingAmount'].mean(),
                'lat': customer_orders['lat'].mean(),
                'lon': customer_orders['lon'].mean(),
                'days_as_customer': (datetime.now() - customer_orders['CreatedOn'].min()).days,
                'lifetime_value_rate': customer_orders['TotalPrice'].sum() / max(1, (datetime.now() - customer_orders['CreatedOn'].min()).days)
            }
            
            customer_features.append(features)
            
        self.customer_embeddings = pd.DataFrame(customer_features)
        return self.customer_embeddings
    
    def create_dynamic_segments(self, min_segment_size=50):
        """Create dynamic customer segments using DBSCAN for varying densities"""
        # Prepare features for clustering
        feature_cols = [
            'total_revenue', 'avg_order_value', 'avg_products_per_order',
            'avg_purchase_interval', 'purchase_regularity', 'seasonal_concentration',
            'lifetime_value_rate'
        ]
        
        X = self.customer_embeddings[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply DBSCAN for dynamic clustering
        dbscan = DBSCAN(eps=0.5, min_samples=min_segment_size)
        clusters = dbscan.fit_predict(X_scaled)
        
        # For customers not in any cluster, use K-means
        outliers = clusters == -1
        if outliers.sum() > 0:
            kmeans = KMeans(n_clusters=3, random_state=42)
            outlier_clusters = kmeans.fit_predict(X_scaled[outliers])
            clusters[outliers] = outlier_clusters + clusters.max() + 1
        
        self.customer_embeddings['segment'] = clusters
        
        # Create segment profiles
        segment_profiles = self.analyze_segments()
        
        return segment_profiles
    
    def analyze_segments(self):
        """Analyze and label customer segments"""
        segment_analysis = self.customer_embeddings.groupby('segment').agg({
            'total_revenue': ['mean', 'sum'],
            'avg_order_value': 'mean',
            'total_orders': 'mean',
            'avg_products_per_order': 'mean',
            'avg_purchase_interval': 'mean',
            'seasonal_concentration': 'mean',
            'IdCustomer': 'count'
        }).round(2)
        
        # Flatten column names
        segment_analysis.columns = ['_'.join(col).strip() for col in segment_analysis.columns.values]
        
        # Create segment labels
        labels = []
        for idx, row in segment_analysis.iterrows():
            characteristics = []
            
            if row['total_revenue_mean'] > segment_analysis['total_revenue_mean'].quantile(0.75):
                characteristics.append("VIP")
            if row['total_orders_mean'] > segment_analysis['total_orders_mean'].quantile(0.75):
                characteristics.append("Loyal")
            if row['avg_purchase_interval_mean'] < segment_analysis['avg_purchase_interval_mean'].quantile(0.25):
                characteristics.append("Frequent")
            if row['seasonal_concentration_mean'] > 0.5:
                characteristics.append("Seasonal")
            if row['avg_products_per_order_mean'] > segment_analysis['avg_products_per_order_mean'].quantile(0.75):
                characteristics.append("Bulk")
                
            label = "-".join(characteristics) if characteristics else "Standard"
            labels.append(label)
            
        segment_analysis['segment_label'] = labels
        
        return segment_analysis
    
    def create_smart_bundles(self):
        """Create intelligent bundles based on segment characteristics"""
        bundles = {}
        
        for segment in self.customer_embeddings['segment'].unique():
            segment_customers = self.customer_embeddings[
                self.customer_embeddings['segment'] == segment
            ]
            segment_orders = self.orders_completed[
                self.orders_completed['IdCustomer'].isin(segment_customers['IdCustomer'])
            ]
            
            # Analyze purchase patterns
            price_distribution = segment_orders['TotalPrice'].describe()
            product_distribution = segment_orders['TotalProducts'].describe()
            
            # Create tiered bundles
            bundles[segment] = {
                'segment_size': len(segment_customers),
                'segment_characteristics': self.analyze_segments().loc[segment, 'segment_label'],
                'bundles': []
            }
            
            # Entry bundle
            if price_distribution['25%'] > 0:
                bundles[segment]['bundles'].append({
                    'tier': 'Entry',
                    'name': f'Starter Pack - {bundles[segment]["segment_characteristics"]}',
                    'target_price': round(price_distribution['25%'] * 0.9, 2),
                    'product_count': int(product_distribution['25%']),
                    'discount': '10%',
                    'cross_sell_potential': 'High',
                    'expected_conversion': self._estimate_conversion(segment, 'entry')
                })
            
            # Core bundle
            bundles[segment]['bundles'].append({
                'tier': 'Core',
                'name': f'Essential Bundle - {bundles[segment]["segment_characteristics"]}',
                'target_price': round(price_distribution['50%'] * 0.85, 2),
                'product_count': int(product_distribution['50%']),
                'discount': '15%',
                'cross_sell_potential': 'Medium',
                'expected_conversion': self._estimate_conversion(segment, 'core')
            })
            
            # Premium bundle
            bundles[segment]['bundles'].append({
                'tier': 'Premium',
                'name': f'Complete Solution - {bundles[segment]["segment_characteristics"]}',
                'target_price': round(price_distribution['75%'] * 0.8, 2),
                'product_count': int(product_distribution['75%']),
                'discount': '20%',
                'cross_sell_potential': 'Low',
                'expected_conversion': self._estimate_conversion(segment, 'premium')
            })
            
            # Seasonal bundle (if applicable)
            if segment_customers['seasonal_concentration'].mean() > 0.4:
                peak_month = segment_customers['preferred_month'].mode()[0]
                bundles[segment]['bundles'].append({
                    'tier': 'Seasonal',
                    'name': f'Season Special - Month {peak_month}',
                    'target_price': round(price_distribution['mean'] * 0.75, 2),
                    'product_count': int(product_distribution['mean'] * 1.2),
                    'discount': '25%',
                    'cross_sell_potential': 'High',
                    'expected_conversion': self._estimate_conversion(segment, 'seasonal'),
                    'best_timing': f'Month {peak_month}'
                })
        
        self.bundle_catalog = bundles
        return bundles
    
    def _estimate_conversion(self, segment, tier):
        """Estimate conversion rate for bundle based on historical data"""
        # Simplified conversion estimation
        base_rates = {'entry': 0.15, 'core': 0.25, 'premium': 0.10, 'seasonal': 0.20}
        
        # Adjust based on segment characteristics
        segment_data = self.customer_embeddings[self.customer_embeddings['segment'] == segment]
        
        adjustment = 1.0
        if segment_data['purchase_regularity'].mean() < 30:  # Regular buyers
            adjustment *= 1.2
        if segment_data['total_revenue'].mean() > self.customer_embeddings['total_revenue'].mean():
            adjustment *= 1.1
            
        return round(base_rates.get(tier, 0.1) * adjustment, 3)
    
    def recommend_bundles_for_customer(self, customer_id, n_recommendations=3):
        """Get personalized bundle recommendations for a specific customer"""
        if customer_id not in self.customer_embeddings['IdCustomer'].values:
            return "Customer not found"
        
        customer_data = self.customer_embeddings[
            self.customer_embeddings['IdCustomer'] == customer_id
        ].iloc[0]
        
        segment = customer_data['segment']
        available_bundles = self.bundle_catalog[segment]['bundles']
        
        # Score bundles based on customer profile
        scored_bundles = []
        for bundle in available_bundles:
            score = 0
            
            # Price affinity
            customer_avg = customer_data['avg_order_value']
            price_ratio = bundle['target_price'] / customer_avg if customer_avg > 0 else 1
            if 0.8 <= price_ratio <= 1.2:
                score += 0.4
            elif 0.6 <= price_ratio <= 1.5:
                score += 0.2
                
            # Product count affinity
            product_ratio = bundle['product_count'] / customer_data['avg_products_per_order']
            if 0.9 <= product_ratio <= 1.3:
                score += 0.3
                
            # Timing score (for seasonal bundles)
            if 'best_timing' in bundle:
                current_month = datetime.now().month
                if f'Month {current_month}' in bundle['best_timing']:
                    score += 0.3
                    
            # Conversion probability
            score += bundle['expected_conversion']
            
            scored_bundles.append({
                'bundle': bundle,
                'score': score,
                'personalization_reason': self._get_recommendation_reason(customer_data, bundle)
            })
        
        # Sort by score and return top N
        scored_bundles.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_bundles[:n_recommendations]
    
    def _get_recommendation_reason(self, customer_data, bundle):
        """Generate explanation for why bundle is recommended"""
        reasons = []
        
        if bundle['tier'] == 'Entry' and customer_data['total_orders'] < 3:
            reasons.append("Perfect for new customers")
        elif bundle['tier'] == 'Premium' and customer_data['total_revenue'] > self.customer_embeddings['total_revenue'].quantile(0.75):
            reasons.append("Exclusive offer for VIP customers")
        elif bundle['tier'] == 'Core':
            reasons.append("Most popular choice in your segment")
        elif bundle['tier'] == 'Seasonal':
            reasons.append("Limited time seasonal offer")
            
        # Price-based reasoning
        price_ratio = bundle['target_price'] / customer_data['avg_order_value'] if customer_data['avg_order_value'] > 0 else 1
        if price_ratio < 0.9:
            reasons.append(f"Save {int((1-price_ratio)*100)}% compared to your usual orders")
        
        return " | ".join(reasons) if reasons else "Recommended based on your purchase history"
    
    def predict_bundle_performance(self):
        """Predict expected performance metrics for each bundle"""
        performance_predictions = {}
        
        for segment, segment_data in self.bundle_catalog.items():
            segment_customers = self.customer_embeddings[
                self.customer_embeddings['segment'] == segment
            ]
            
            performance_predictions[segment] = {
                'segment_label': segment_data['segment_characteristics'],
                'bundles': []
            }
            
            for bundle in segment_data['bundles']:
                # Estimate metrics
                expected_adoption = bundle['expected_conversion'] * len(segment_customers)
                expected_revenue = expected_adoption * bundle['target_price']
                
                # Calculate uplift potential
                avg_customer_value = segment_customers['avg_order_value'].mean()
                uplift_per_customer = max(0, bundle['target_price'] - avg_customer_value)
                total_uplift = uplift_per_customer * expected_adoption
                
                performance_predictions[segment]['bundles'].append({
                    'bundle_name': bundle['name'],
                    'expected_adoption': int(expected_adoption),
                    'expected_revenue': round(expected_revenue, 2),
                    'revenue_uplift': round(total_uplift, 2),
                    'roi_estimate': round((total_uplift / bundle['target_price']) * 100, 1) if bundle['target_price'] > 0 else 0
                })
        
        return performance_predictions
    
    def optimize_bundle_pricing(self, price_elasticity=0.2):
        """Optimize bundle pricing based on demand elasticity"""
        optimized_bundles = {}
        
        for segment, segment_data in self.bundle_catalog.items():
            segment_orders = self.orders_completed[
                self.orders_completed['IdCustomer'].isin(
                    self.customer_embeddings[self.customer_embeddings['segment'] == segment]['IdCustomer']
                )
            ]
            
            optimized_bundles[segment] = {
                'segment_characteristics': segment_data['segment_characteristics'],
                'bundles': []
            }
            
            for bundle in segment_data['bundles']:
                # Price optimization based on elasticity
                base_price = bundle['target_price']
                
                # Test different price points
                price_tests = []
                for price_multiplier in [0.9, 0.95, 1.0, 1.05, 1.1]:
                    test_price = base_price * price_multiplier
                    
                    # Estimate demand change based on elasticity
                    price_change = (price_multiplier - 1.0)
                    demand_change = -price_change * price_elasticity
                    adjusted_conversion = bundle['expected_conversion'] * (1 + demand_change)
                    
                    # Calculate expected revenue
                    segment_size = segment_data['segment_size']
                    expected_revenue = test_price * adjusted_conversion * segment_size
                    
                    price_tests.append({
                        'price': round(test_price, 2),
                        'conversion': round(adjusted_conversion, 3),
                        'revenue': round(expected_revenue, 2)
                    })
                
                # Find optimal price
                optimal = max(price_tests, key=lambda x: x['revenue'])
                
                optimized_bundle = bundle.copy()
                optimized_bundle['optimized_price'] = optimal['price']
                optimized_bundle['price_adjustment'] = round((optimal['price'] / base_price - 1) * 100, 1)
                optimized_bundle['expected_revenue_increase'] = round(
                    (optimal['revenue'] - base_price * bundle['expected_conversion'] * segment_data['segment_size']), 2
                )
                
                optimized_bundles[segment]['bundles'].append(optimized_bundle)
        
        return optimized_bundles
    
    def create_cross_sell_matrix(self):
        """Create a matrix of cross-selling opportunities between bundles"""
        cross_sell_opportunities = []
        
        # Analyze customer progression patterns
        customers_with_multiple_orders = self.orders_completed.groupby('IdCustomer').filter(
            lambda x: len(x) >= 2
        )['IdCustomer'].unique()
        
        for customer_id in customers_with_multiple_orders[:1000]:  # Sample for efficiency
            customer_orders = self.orders_completed[
                self.orders_completed['IdCustomer'] == customer_id
            ].sort_values('CreatedOn')
            
            if len(customer_orders) < 2:
                continue
                
            # Track order progression
            for i in range(len(customer_orders) - 1):
                order1 = customer_orders.iloc[i]
                order2 = customer_orders.iloc[i + 1]
                
                value_progression = order2['TotalPrice'] / order1['TotalPrice'] if order1['TotalPrice'] > 0 else 1
                product_progression = order2['TotalProducts'] / order1['TotalProducts'] if order1['TotalProducts'] > 0 else 1
                time_between = (order2['CreatedOn'] - order1['CreatedOn']).days
                
                cross_sell_opportunities.append({
                    'value_progression': value_progression,
                    'product_progression': product_progression,
                    'days_between': time_between
                })
        
        cross_sell_df = pd.DataFrame(cross_sell_opportunities)
        
        # Create recommendations based on patterns
        cross_sell_insights = {
            'avg_value_increase': round(cross_sell_df['value_progression'].mean(), 2),
            'avg_product_increase': round(cross_sell_df['product_progression'].mean(), 2),
            'optimal_follow_up_days': int(cross_sell_df['days_between'].median()),
            'upgrade_patterns': {
                'entry_to_core': len(cross_sell_df[(cross_sell_df['value_progression'] > 1.2) & 
                                                   (cross_sell_df['value_progression'] < 2.0)]) / len(cross_sell_df),
                'core_to_premium': len(cross_sell_df[cross_sell_df['value_progression'] > 2.0]) / len(cross_sell_df),
                'repeat_same_tier': len(cross_sell_df[(cross_sell_df['value_progression'] > 0.8) & 
                                                      (cross_sell_df['value_progression'] < 1.2)]) / len(cross_sell_df)
            }
        }
        
        return cross_sell_insights
    
    def generate_marketing_insights(self):
        """Generate actionable marketing insights for bundle campaigns"""
        insights = {
            'segment_insights': [],
            'timing_recommendations': {},
            'channel_recommendations': {}
        }
        
        # Analyze each segment
        for segment in self.customer_embeddings['segment'].unique():
            segment_data = self.customer_embeddings[self.customer_embeddings['segment'] == segment]
            segment_orders = self.orders_completed[
                self.orders_completed['IdCustomer'].isin(segment_data['IdCustomer'])
            ]
            
            # Geographic concentration
            top_countries = segment_orders['DeliveryCountry'].value_counts().head(3)
            
            # Timing patterns
            order_hours = pd.to_datetime(segment_orders['CreatedOn']).dt.hour
            peak_hours = order_hours.value_counts().head(3).index.tolist()
            
            order_days = pd.to_datetime(segment_orders['CreatedOn']).dt.dayofweek
            peak_days = order_days.value_counts().head(3).index.tolist()
            
            insights['segment_insights'].append({
                'segment': segment,
                'label': self.bundle_catalog[segment]['segment_characteristics'],
                'size': len(segment_data),
                'avg_ltv': round(segment_data['total_revenue'].mean(), 2),
                'top_countries': top_countries.to_dict(),
                'peak_ordering_hours': peak_hours,
                'peak_ordering_days': peak_days,
                'recommended_campaign_type': self._recommend_campaign_type(segment_data)
            })
        
        # Overall timing recommendations
        all_orders = pd.to_datetime(self.orders_completed['CreatedOn'])
        insights['timing_recommendations'] = {
            'best_months': all_orders.dt.month.value_counts().head(3).index.tolist(),
            'best_days_of_week': all_orders.dt.dayofweek.value_counts().head(3).index.tolist(),
            'best_hours': all_orders.dt.hour.value_counts().head(3).index.tolist()
        }
        
        return insights
    
    def _recommend_campaign_type(self, segment_data):
        """Recommend campaign type based on segment characteristics"""
        if segment_data['purchase_regularity'].mean() < 20:
            return "Loyalty program with recurring benefits"
        elif segment_data['avg_order_value'].mean() > self.customer_embeddings['avg_order_value'].quantile(0.75):
            return "VIP exclusive previews and early access"
        elif segment_data['seasonal_concentration'].mean() > 0.5:
            return "Seasonal campaigns with time-limited offers"
        else:
            return "Volume discounts and bulk purchase incentives"
    
    def export_recommendations(self, output_format='json'):
        """Export all recommendations and insights"""
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'summary_statistics': {
                'total_customers_analyzed': len(self.customer_embeddings),
                'total_segments': len(self.customer_embeddings['segment'].unique()),
                'total_bundles_created': sum(len(seg['bundles']) for seg in self.bundle_catalog.values())
            },
            'segment_profiles': self.analyze_segments().to_dict(),
            'bundle_catalog': self.bundle_catalog,
            'performance_predictions': self.predict_bundle_performance(),
            'optimized_pricing': self.optimize_bundle_pricing(),
            'cross_sell_insights': self.create_cross_sell_matrix(),
            'marketing_insights': self.generate_marketing_insights()
        }
        
        if output_format == 'json':
            import json
            return json.dumps(export_data, indent=2, default=str)
        else:
            return export_data
    
    def train_complete_system(self, orders_filepath, products_filepath=None):
        """Complete training pipeline"""
        print("=== Advanced Bundle Recommendation System Training ===\n")
        
        # Load data
        print("1. Loading data...")
        self.load_order_data(orders_filepath, products_filepath)
        
        # Create customer embeddings
        print("2. Creating customer embeddings...")
        self.create_customer_embedding()
        
        # Create RFM analysis
        print("3. Performing RFM analysis...")
        rfm_data = self.create_rfm_features()
        print(f"   - Created RFM scores for {len(rfm_data)} customers")
        
        # Segment customers
        print("4. Creating dynamic customer segments...")
        segment_profiles = self.create_dynamic_segments()
        print(f"   - Identified {len(segment_profiles)} customer segments")
        print("\nSegment Profiles:")
        print(segment_profiles[['segment_label', 'IdCustomer_count']])
        
        # Create bundles
        print("\n5. Creating intelligent bundle recommendations...")
        bundles = self.create_smart_bundles()
        total_bundles = sum(len(seg['bundles']) for seg in bundles.values())
        print(f"   - Created {total_bundles} bundles across all segments")
        
        # Optimize pricing
        print("\n6. Optimizing bundle pricing...")
        optimized = self.optimize_bundle_pricing()
        
        # Generate insights
        print("\n7. Generating marketing insights...")
        insights = self.generate_marketing_insights()
        
        print("\n=== Training Complete ===")
        print(f"\nKey Results:")
        print(f"- Analyzed {len(self.customer_embeddings)} customers")
        print(f"- Created {len(segment_profiles)} distinct segments")
        print(f"- Generated {total_bundles} personalized bundles")
        print(f"- Average expected revenue increase: {self._calculate_avg_revenue_increase()}%")
        
        return segment_profiles
    
    def _calculate_avg_revenue_increase(self):
        """Calculate average expected revenue increase from bundles"""
        predictions = self.predict_bundle_performance()
        total_uplift = sum(
            bundle['revenue_uplift'] 
            for segment in predictions.values() 
            for bundle in segment['bundles']
        )
        current_revenue = self.orders_completed['TotalPrice'].sum()
        return round((total_uplift / current_revenue) * 100, 1)


# Example usage
if __name__ == "__main__":
    # Initialize the system
    system = AdvancedBundleRecommendationSystem()
    
    # Train the complete system
    system.train_complete_system('data/weather_api/orders_with_coords_final.csv')
    
    # Get recommendations for a specific customer
    # customer_id = 16262042
    # recommendations = system.recommend_bundles_for_customer(customer_id)
    # print(f"\nPersonalized recommendations for customer {customer_id}:")
    # for rec in recommendations:
    #     print(f"- {rec['bundle']['name']}: €{rec['bundle']['target_price']}")
    #     print(f"  Reason: {rec['personalization_reason']}")
    #     print(f"  Score: {rec['score']:.2f}\n")
    
    # Export all recommendations
    # export_data = system.export_recommendations()
    # with open('bundle_recommendations.json', 'w') as f:
    #     f.write(export_data)