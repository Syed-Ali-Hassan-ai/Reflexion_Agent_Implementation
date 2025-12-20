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
        def calculate_average_sales(num_weeks: int = 4) -> str:
            """
            Calculate average sales over the last N weeks.
            Use this to get a baseline understanding of recent sales performance.

            Args:
                num_weeks: Number of recent weeks to average (default: 4)

            Returns:
                String describing the average sales
            """
            try:
                num_weeks = int(num_weeks)
                if len(self.historical_data) == 0:
                    return "No historical data available."

                recent_data = self.historical_data.tail(num_weeks)
                avg_sales = recent_data[self.sales_column].mean()

                return f"Average sales over the last {num_weeks} weeks: ${avg_sales:,.2f}"
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

                return " ".join(insights) if insights else "No clear seasonal patterns detected."

            except Exception as e:
                return f"Error checking seasonality: {str(e)}"

        @tool
        def get_historical_data(num_weeks: int = 8) -> str:
            """
            Get detailed historical sales data for the last N weeks.
            Use this to see the actual week-by-week sales figures.

            Args:
                num_weeks: Number of recent weeks to retrieve (default: 8)

            Returns:
                String with formatted historical data
            """
            try:
                num_weeks = int(num_weeks)
                if len(self.historical_data) == 0:
                    return "No historical data available."

                recent_data = self.historical_data.tail(num_weeks)

                result = f"Last {min(num_weeks, len(recent_data))} weeks of sales data:\n"
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
        def make_prediction(predicted_sales: float, reasoning: str = "") -> str:
            """
            Make your final sales prediction.
            Use this as the last step after analyzing the data.

            Args:
                predicted_sales: Your predicted sales amount (in dollars)
                reasoning: Brief explanation of your reasoning (optional)

            Returns:
                Confirmation of prediction
            """
            try:
                predicted_sales = float(predicted_sales)
                result = f"Prediction recorded: ${predicted_sales:,.2f}"
                if reasoning:
                    result += f"\nReasoning: {reasoning}"
                return result
            except Exception as e:
                return f"Error making prediction: {str(e)}"

        return [
            calculate_average_sales,
            check_seasonality,
            get_historical_data,
            analyze_trends,
            make_prediction
        ]
