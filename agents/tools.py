"""
Tools for sales forecasting agents
"""
import pandas as pd
import numpy as np
from langchain.tools import tool
from typing import Dict, Any


class SalesAnalysisTools:
    """Tools for analyzing sales data"""

    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize tools with historical data

        Args:
            historical_data: DataFrame with historical sales data
        """
        self.historical_data = historical_data
        self.sales_column = self._find_sales_column()

    def _find_sales_column(self) -> str:
        """Find the sales column in the dataframe"""
        for col in self.historical_data.columns:
            if 'sales' in col.lower():
                return col
        return 'weekly_sales'  # default

    def get_tools(self):
        """Get list of tools for the agent"""

        @tool
        def calculate_average_sales(num_weeks: str = "4") -> str:
            """
            Calculate average sales over the last N weeks.
            Use this to get a baseline understanding of recent sales performance.

            Args:
                num_weeks: Number of recent weeks to average (e.g., "4", "8", "12")

            Returns:
                String describing the average sales
            """
            try:
                # Parse input - handle various formats
                weeks_str = str(num_weeks).strip()
                # Extract first number found
                import re
                numbers = re.findall(r'(\d+)', weeks_str)
                if numbers:
                    num = int(numbers[0])
                else:
                    num = 4  # default

                if len(self.historical_data) == 0:
                    return "No historical data available."

                recent_data = self.historical_data.tail(num)
                avg_sales = recent_data[self.sales_column].mean()

                return f"Average sales over the last {num} weeks: ${avg_sales:,.2f}"
            except Exception as e:
                return f"Error calculating average: {str(e)}"

        @tool
        def check_seasonality(query: str = "") -> str:
            """
            Check for seasonal patterns in the sales data.
            Use this to identify if there are recurring patterns (weekly, monthly, holiday effects).

            Args:
                query: Optional query parameter (not used, for compatibility)

            Returns:
                String describing seasonal patterns found
            """
            try:
                if len(self.historical_data) < 8:
                    return "Not enough data to determine seasonality (need at least 8 weeks)."

                sales = self.historical_data[self.sales_column]

                # Check if there's a date column
                date_col = None
                for col in self.historical_data.columns:
                    if 'date' in col.lower() or 'week' in col.lower():
                        date_col = col
                        break

                insights = []

                # Calculate coefficient of variation
                cv = sales.std() / sales.mean() if sales.mean() != 0 else 0

                if cv < 0.1:
                    insights.append("Sales are very stable with minimal variation.")
                elif cv < 0.3:
                    insights.append("Sales show moderate variation.")
                else:
                    insights.append("Sales show high variation, indicating strong seasonal or promotional effects.")

                # Check for trend
                if len(sales) >= 4:
                    recent_avg = sales.tail(4).mean()
                    older_avg = sales.head(4).mean()
                    pct_change = ((recent_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0

                    if abs(pct_change) > 10:
                        if pct_change > 0:
                            insights.append(f"Strong upward trend detected ({pct_change:.1f}% increase).")
                        else:
                            insights.append(f"Strong downward trend detected ({abs(pct_change):.1f}% decrease).")

                # Check for holiday indicator
                if 'is_holiday' in self.historical_data.columns:
                    holiday_avg = self.historical_data[self.historical_data['is_holiday'] == 1][self.sales_column].mean()
                    regular_avg = self.historical_data[self.historical_data['is_holiday'] == 0][self.sales_column].mean()

                    if pd.notna(holiday_avg) and pd.notna(regular_avg) and regular_avg > 0:
                        holiday_boost = ((holiday_avg - regular_avg) / regular_avg * 100)
                        if holiday_boost > 5:
                            insights.append(f"Holiday weeks show {holiday_boost:.1f}% higher sales on average.")

                # CRITICAL: Check for post-holiday pattern
                if 'is_post_holiday' in self.historical_data.columns:
                    post_holiday_avg = self.historical_data[self.historical_data['is_post_holiday'] == 1][self.sales_column].mean()
                    regular_avg = self.historical_data[self.historical_data['is_post_holiday'] == 0][self.sales_column].mean()

                    if pd.notna(post_holiday_avg) and pd.notna(regular_avg) and regular_avg > 0:
                        post_holiday_change = ((post_holiday_avg - regular_avg) / regular_avg * 100)
                        if abs(post_holiday_change) > 5:
                            if post_holiday_change < 0:
                                insights.append(f"WARNING: Weeks AFTER holidays show {abs(post_holiday_change):.1f}% LOWER sales (post-holiday slump).")
                            else:
                                insights.append(f"Weeks after holidays show {post_holiday_change:.1f}% higher sales.")

                # Check for promotions
                if 'is_promotion' in self.historical_data.columns:
                    promo_avg = self.historical_data[self.historical_data['is_promotion'] == 1][self.sales_column].mean()
                    regular_avg = self.historical_data[self.historical_data['is_promotion'] == 0][self.sales_column].mean()

                    if pd.notna(promo_avg) and pd.notna(regular_avg) and regular_avg > 0:
                        promo_boost = ((promo_avg - regular_avg) / regular_avg * 100)
                        if promo_boost > 5:
                            insights.append(f"Promotion weeks show {promo_boost:.1f}% higher sales.")

                # Check for stock-outs
                if 'is_stockout' in self.historical_data.columns:
                    stockout_avg = self.historical_data[self.historical_data['is_stockout'] == 1][self.sales_column].mean()
                    regular_avg = self.historical_data[self.historical_data['is_stockout'] == 0][self.sales_column].mean()

                    if pd.notna(stockout_avg) and pd.notna(regular_avg) and regular_avg > 0:
                        stockout_impact = ((stockout_avg - regular_avg) / regular_avg * 100)
                        if stockout_impact < -5:
                            insights.append(f"Stock-out weeks show {abs(stockout_impact):.1f}% LOWER sales due to inventory issues.")

                return " ".join(insights) if insights else "No clear seasonal patterns detected."

            except Exception as e:
                return f"Error checking seasonality: {str(e)}"

        @tool
        def get_historical_data(num_weeks: str = "8") -> str:
            """
            Get detailed historical sales data for the last N weeks.
            Use this to see the actual week-by-week sales figures.

            Args:
                num_weeks: Number of recent weeks to retrieve (e.g., "8", "10", "12")

            Returns:
                String with formatted historical data
            """
            try:
                # Parse input - handle various formats
                weeks_str = str(num_weeks).strip()
                # Extract first number found
                import re
                numbers = re.findall(r'(\d+)', weeks_str)
                if numbers:
                    num = int(numbers[0])
                else:
                    num = 8  # default

                if len(self.historical_data) == 0:
                    return "No historical data available."

                recent_data = self.historical_data.tail(num)

                result = f"Last {min(num, len(recent_data))} weeks of sales data:\n"
                for idx, row in recent_data.iterrows():
                    sales_val = row[self.sales_column]
                    result += f"  Week {idx + 1}: ${sales_val:,.2f}\n"

                return result.strip()
            except Exception as e:
                return f"Error retrieving historical data: {str(e)}"

        @tool
        def analyze_trends(query: str = "") -> str:
            """
            Analyze trends in the sales data.
            Use this to identify if sales are growing, declining, or stable.

            Args:
                query: Optional query parameter (not used, for compatibility)

            Returns:
                String describing the trend analysis
            """
            try:
                if len(self.historical_data) < 4:
                    return "Not enough data to analyze trends (need at least 4 weeks)."

                sales = self.historical_data[self.sales_column]

                # Simple linear trend
                x = np.arange(len(sales))
                y = sales.values

                # Calculate slope using simple linear regression
                mean_x = x.mean()
                mean_y = y.mean()

                numerator = ((x - mean_x) * (y - mean_y)).sum()
                denominator = ((x - mean_x) ** 2).sum()

                if denominator != 0:
                    slope = numerator / denominator
                    weekly_change = slope
                    pct_change = (slope / mean_y * 100) if mean_y != 0 else 0

                    if abs(pct_change) < 1:
                        trend = "relatively stable"
                    elif pct_change > 5:
                        trend = "strongly growing"
                    elif pct_change > 0:
                        trend = "growing"
                    elif pct_change < -5:
                        trend = "strongly declining"
                    else:
                        trend = "declining"

                    return f"Trend analysis: Sales are {trend}. Average weekly change: ${weekly_change:,.2f} ({pct_change:.2f}%)."
                else:
                    return "Unable to calculate trend."

            except Exception as e:
                return f"Error analyzing trends: {str(e)}"

        @tool
        def make_prediction(predicted_sales: str) -> str:
            """
            Make your final sales prediction.
            Use this as the last step after analyzing the data.

            Args:
                predicted_sales: Your predicted sales amount in dollars (just the number, e.g., "50000" or "50000.50")

            Returns:
                Confirmation of prediction

            Example:
                make_prediction("75000")
                make_prediction("75123.45")
            """
            try:
                # Handle different input formats
                sales_str = str(predicted_sales).strip()

                # Remove dollar signs and commas if present
                sales_str = sales_str.replace('$', '').replace(',', '')

                # Extract just the number (in case there's extra text)
                import re
                numbers = re.findall(r'([0-9]+\.?[0-9]*)', sales_str)
                if numbers:
                    sales_value = float(numbers[0])
                else:
                    sales_value = float(sales_str)

                result = f"Prediction recorded: ${sales_value:,.2f}"
                return result
            except (ValueError, AttributeError) as e:
                return f"Error making prediction: Could not parse '{predicted_sales}' as a number. Please provide just the numeric value (e.g., '50000')."
            except Exception as e:
                return f"Error making prediction: {str(e)}"

        return [
            calculate_average_sales,
            check_seasonality,
            get_historical_data,
            analyze_trends,
            make_prediction
        ]
