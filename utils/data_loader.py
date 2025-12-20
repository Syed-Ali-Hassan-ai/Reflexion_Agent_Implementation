"""
Data loader for Walmart sales dataset
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import config


class WalmartDataLoader:
    """Loads and preprocesses Walmart sales dataset"""

    def __init__(self):
        self.df = None
        self.test_cases = []

    def load_data(self, use_sample=False) -> pd.DataFrame:
        """
        Load Walmart sales dataset

        Args:
            use_sample: If True, create sample data instead of loading from file

        Returns:
            DataFrame with sales data
        """
        if use_sample:
            return self._create_sample_data()

        try:
            # Try to load from kagglehub
            import kagglehub
            path = kagglehub.dataset_download(config.KAGGLE_DATASET)

            # Find the CSV file in the downloaded path
            csv_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))

            if csv_files:
                self.df = pd.read_csv(csv_files[0])
            else:
                print("No CSV file found in downloaded dataset. Using sample data.")
                self.df = self._create_sample_data()

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using sample data instead.")
            self.df = self._create_sample_data()

        # Standardize column names
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Check if required columns exist for department-level forecasting
        has_dept = any('dept' in col for col in self.df.columns)
        has_store = any('store' in col for col in self.df.columns)
        has_sales = any('sales' in col for col in self.df.columns)

        if not has_dept:
            print("=" * 60)
            print("INFO: Kaggle Walmart dataset doesn't include department data.")
            print("Using generated sample data with department breakdowns instead.")
            print("Sample data includes realistic trends and seasonality.")
            print("=" * 60)
            self.df = self._create_sample_data()
        elif not (has_store and has_sales):
            print("Warning: Missing required columns. Using sample data.")
            self.df = self._create_sample_data()

        # Convert date column if it exists
        date_columns = ['date', 'week', 'week_date']
        for col in date_columns:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.df = self.df.sort_values(col)
                    break
                except:
                    pass

        return self.df

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample Walmart sales data with complex patterns for demonstration"""
        # Use time-based seed for variation across runs, but deterministic within run
        import time
        seed = int(time.time() * 1000) % (2**32)
        np.random.seed(seed)

        stores = [1, 2, 3, 4, 5]
        departments = [1, 2, 3, 4, 5, 6, 7, 8]
        weeks = pd.date_range(start='2020-01-01', periods=104, freq='W')

        data = []
        for store in stores:
            for dept in departments:
                base_sales = np.random.uniform(15000, 45000)

                # Department-specific volatility
                dept_volatility = 0.15 if dept in [1, 2] else 0.08

                was_holiday_last_week = False
                was_promo_last_week = False

                for i, week in enumerate(weeks):
                    # Base trend
                    trend = base_sales * (1 + 0.002 * i)

                    # Strong seasonality
                    month = week.month
                    day = week.day
                    seasonal = 1.0

                    is_holiday = 0

                    # Pre-holiday boost (week BEFORE major holidays)
                    if (month == 11 and day >= 15) or (month == 12 and day >= 15):
                        seasonal = 1.5  # Major boost before Christmas
                        is_holiday = 1
                    elif month == 12 and day <= 7:
                        seasonal = 1.4  # Early December
                    elif month == 11 and day >= 1:
                        seasonal = 1.25  # November (Thanksgiving prep)
                    elif month in [6, 7]:
                        seasonal = 1.15  # Summer
                    elif month in [1, 2]:
                        seasonal = 0.85  # Post-holiday slump

                    # CRITICAL PATTERN: Post-holiday drop
                    # Week AFTER holiday has LOWER sales than average
                    if was_holiday_last_week:
                        seasonal = 0.65  # 35% DROP after holiday week
                        is_holiday = 0

                    # Promotion weeks (random, ~10% of weeks)
                    is_promo = np.random.random() < 0.1
                    if is_promo and not was_holiday_last_week:
                        seasonal *= 1.35  # 35% boost during promos

                    # Stock-out events (random, ~5% of weeks)
                    is_stockout = np.random.random() < 0.05
                    if is_stockout:
                        seasonal *= 0.55  # 45% drop due to stock issues

                    # End-of-month effect (people paid, spend more)
                    if day >= 25:
                        seasonal *= 1.08

                    # Random noise (department-specific)
                    noise = np.random.normal(1.0, dept_volatility)

                    weekly_sales = trend * seasonal * noise

                    data.append({
                        'store': store,
                        'dept': dept,
                        'date': week,
                        'weekly_sales': max(1000, weekly_sales),  # Minimum sales
                        'is_holiday': is_holiday,
                        'is_promotion': 1 if is_promo else 0,
                        'is_stockout': 1 if is_stockout else 0,
                        'is_post_holiday': 1 if was_holiday_last_week else 0
                    })

                    # Track state for next week
                    was_holiday_last_week = (is_holiday == 1)
                    was_promo_last_week = is_promo

        return pd.DataFrame(data)

    def get_store_dept_data(self, store: int, dept: int) -> pd.DataFrame:
        """Get data for specific store and department"""
        if self.df is None:
            self.load_data()

        # Try different possible column names (case-insensitive)
        store_col = None
        dept_col = None

        for col in self.df.columns:
            if 'store' in col.lower():
                store_col = col
            if 'dept' in col.lower():
                dept_col = col

        if store_col and dept_col:
            return self.df[(self.df[store_col] == store) & (self.df[dept_col] == dept)].copy()
        else:
            return pd.DataFrame()

    def generate_test_cases(self, num_cases: int = 10) -> List[Dict]:
        """
        Generate test cases for evaluation

        Args:
            num_cases: Number of test cases to generate

        Returns:
            List of test case dictionaries
        """
        if self.df is None:
            self.load_data()

        test_cases = []

        # Get unique stores and departments - with robust column detection
        try:
            store_col = [col for col in self.df.columns if 'store' in col.lower()][0]
        except IndexError:
            print(f"Warning: No 'store' column found. Available columns: {list(self.df.columns)}")
            print("Falling back to sample data...")
            self.df = self._create_sample_data()
            return self.generate_test_cases(num_cases)

        try:
            dept_col = [col for col in self.df.columns if 'dept' in col.lower()][0]
        except IndexError:
            print(f"Warning: No 'dept' column found. Available columns: {list(self.df.columns)}")
            print("Falling back to sample data...")
            self.df = self._create_sample_data()
            return self.generate_test_cases(num_cases)

        try:
            date_col = [col for col in self.df.columns if 'date' in col.lower() or 'week' in col.lower()][0]
        except IndexError:
            print(f"Warning: No 'date' or 'week' column found. Available columns: {list(self.df.columns)}")
            print("Falling back to sample data...")
            self.df = self._create_sample_data()
            return self.generate_test_cases(num_cases)

        try:
            sales_col = [col for col in self.df.columns if 'sales' in col.lower()][0]
        except IndexError:
            print(f"Warning: No 'sales' column found. Available columns: {list(self.df.columns)}")
            print("Falling back to sample data...")
            self.df = self._create_sample_data()
            return self.generate_test_cases(num_cases)

        stores = self.df[store_col].unique()
        depts = self.df[dept_col].unique()

        # Use time-based seed for varied test cases
        import time
        seed = int(time.time() * 1000) % (2**32)
        np.random.seed(seed)

        for i in range(num_cases):
            # Random store and department
            store = np.random.choice(stores)
            dept = np.random.choice(depts)

            # Get data for this store/dept
            subset = self.df[(self.df[store_col] == store) & (self.df[dept_col] == dept)].copy()
            subset = subset.sort_values(date_col)

            if len(subset) < config.MIN_HISTORICAL_WEEKS + 1:
                continue

            # Use last week as test, everything before as context
            test_week_idx = len(subset) - 1 - (i % 5)  # Vary which week we test
            if test_week_idx < config.MIN_HISTORICAL_WEEKS:
                test_week_idx = len(subset) - 1

            test_row = subset.iloc[test_week_idx]
            historical_data = subset.iloc[:test_week_idx]

            test_case = {
                'id': i + 1,
                'store': int(store),
                'department': int(dept),
                'test_week': test_row[date_col],
                'actual_sales': float(test_row[sales_col]),
                'historical_data': historical_data,
                'description': f"Predict sales for Store {int(store)}, Department {int(dept)}, Week {test_row[date_col].strftime('%Y-%m-%d') if hasattr(test_row[date_col], 'strftime') else test_row[date_col]}"
            }

            test_cases.append(test_case)

        self.test_cases = test_cases
        return test_cases

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the dataset"""
        if self.df is None:
            self.load_data()

        try:
            sales_col = [col for col in self.df.columns if 'sales' in col.lower()][0]
            store_col = [col for col in self.df.columns if 'store' in col.lower()][0]
            dept_col = [col for col in self.df.columns if 'dept' in col.lower()][0]

            return {
                'total_records': len(self.df),
                'num_stores': self.df[store_col].nunique(),
                'num_departments': self.df[dept_col].nunique(),
                'avg_sales': float(self.df[sales_col].mean()),
                'min_sales': float(self.df[sales_col].min()),
                'max_sales': float(self.df[sales_col].max()),
            }
        except IndexError:
            # If columns not found, return basic stats
            return {
                'total_records': len(self.df),
                'num_stores': 0,
                'num_departments': 0,
                'avg_sales': 0.0,
                'min_sales': 0.0,
                'max_sales': 0.0,
            }
