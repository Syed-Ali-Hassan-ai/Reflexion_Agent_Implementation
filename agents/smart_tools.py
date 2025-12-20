"""
Enhanced smart tools for sales forecasting with sophisticated analysis
"""
import pandas as pd
import numpy as np
from langchain.tools import tool
from typing import Dict, Any
from scipy import stats
from utils.cache import get_lru_cache


class SmartSalesTools:
    """Enhanced tools with statistical analysis and pattern detection"""

    def __init__(self, historical_data: pd.DataFrame):
        """Initialize with historical data"""
        self.historical_data = historical_data
        self.sales_column = self._find_sales_column()
        self.cache = get_lru_cache()

    def _find_sales_column(self) -> str:
        """Find sales column"""
        for col in self.historical_data.columns:
            if 'sales' in col.lower():
                return col
        return 'weekly_sales'

    def get_tools(self):
        """Get enhanced tool list"""

        @tool
        def calculate_average_sales(num_weeks: str = "4") -> str:
            """
            Calculate average sales with statistical confidence intervals.

            Args:
                num_weeks: Number of recent weeks (e.g., "4", "8", "12")

            Returns:
                Average sales with confidence interval and volatility
            """
            try:
                import re
                numbers = re.findall(r'(\d+)', str(num_weeks).strip())
                num = int(numbers[0]) if numbers else 4

                if len(self.historical_data) == 0:
                    return "No historical data available."

                recent_data = self.historical_data.tail(num)
                sales_values = recent_data[self.sales_column].values

                avg = np.mean(sales_values)
                std = np.std(sales_values)
                volatility_pct = (std / avg * 100) if avg > 0 else 0

                # 95% confidence interval
                ci = 1.96 * std / np.sqrt(len(sales_values))

                result = f"Average sales over last {num} weeks: ${avg:,.2f}\n"
                result += f"Range: ${avg-ci:,.2f} to ${avg+ci:,.2f} (95% confidence)\n"
                result += f"Volatility: {volatility_pct:.1f}% (std: ${std:,.2f})"

                return result

            except Exception as e:
                return f"Error: {str(e)}"

        @tool
        def check_seasonality(query: str = "") -> str:
            """
            Advanced seasonality detection with multiple pattern recognition.

            Detects: holidays, post-holiday slumps, promotions, stock-outs,
            trends, and cyclical patterns.

            Returns:
                Comprehensive seasonality analysis
            """
            try:
                if len(self.historical_data) < 8:
                    return "Need at least 8 weeks of data for seasonality analysis."

                sales = self.historical_data[self.sales_column].values
                insights = []

                # Volatility analysis
                cv = np.std(sales) / np.mean(sales) if np.mean(sales) > 0 else 0
                if cv < 0.1:
                    insights.append("📊 Sales are VERY STABLE (low volatility)")
                elif cv < 0.3:
                    insights.append("📊 Sales show MODERATE variation")
                else:
                    insights.append("⚠️ Sales show HIGH variation - strong seasonal effects")

                # Trend detection with statistical significance
                x = np.arange(len(sales))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, sales)

                if p_value < 0.05:  # Statistically significant trend
                    pct_change = (slope / np.mean(sales) * 100) if np.mean(sales) > 0 else 0
                    if abs(pct_change) > 5:
                        direction = "GROWING" if pct_change > 0 else "DECLINING"
                        insights.append(f"📈 Strong {direction} trend: {abs(pct_change):.1f}% per week (p={p_value:.3f})")

                # Holiday effects
                if 'is_holiday' in self.historical_data.columns:
                    holiday_avg = self.historical_data[self.historical_data['is_holiday'] == 1][self.sales_column].mean()
                    regular_avg = self.historical_data[self.historical_data['is_holiday'] == 0][self.sales_column].mean()

                    if pd.notna(holiday_avg) and pd.notna(regular_avg) and regular_avg > 0:
                        boost = ((holiday_avg - regular_avg) / regular_avg * 100)
                        if abs(boost) > 5:
                            insights.append(f"🎄 HOLIDAY WEEKS: {boost:+.1f}% vs normal (${holiday_avg:,.0f} vs ${regular_avg:,.0f})")

                # POST-HOLIDAY SLUMP (CRITICAL!)
                if 'is_post_holiday' in self.historical_data.columns:
                    post_avg = self.historical_data[self.historical_data['is_post_holiday'] == 1][self.sales_column].mean()
                    regular_avg = self.historical_data[self.historical_data['is_post_holiday'] == 0][self.sales_column].mean()

                    if pd.notna(post_avg) and pd.notna(regular_avg) and regular_avg > 0:
                        change = ((post_avg - regular_avg) / regular_avg * 100)
                        if change < -5:
                            insights.append(f"🚨 POST-HOLIDAY SLUMP: {change:.1f}% LOWER sales (${post_avg:,.0f} vs ${regular_avg:,.0f})")
                            insights.append(f"   ⚠️ Week AFTER holiday shows {abs(change):.0f}% DROP - adjust predictions DOWN!")

                # Promotions
                if 'is_promotion' in self.historical_data.columns:
                    promo_avg = self.historical_data[self.historical_data['is_promotion'] == 1][self.sales_column].mean()
                    regular_avg = self.historical_data[self.historical_data['is_promotion'] == 0][self.sales_column].mean()

                    if pd.notna(promo_avg) and pd.notna(regular_avg) and regular_avg > 0:
                        boost = ((promo_avg - regular_avg) / regular_avg * 100)
                        if boost > 5:
                            insights.append(f"🎯 PROMOTION WEEKS: +{boost:.1f}% boost (${promo_avg:,.0f} vs ${regular_avg:,.0f})")

                # Stock-outs
                if 'is_stockout' in self.historical_data.columns:
                    stockout_avg = self.historical_data[self.historical_data['is_stockout'] == 1][self.sales_column].mean()
                    regular_avg = self.historical_data[self.historical_data['is_stockout'] == 0][self.sales_column].mean()

                    if pd.notna(stockout_avg) and pd.notna(regular_avg) and regular_avg > 0:
                        impact = ((stockout_avg - regular_avg) / regular_avg * 100)
                        if impact < -5:
                            insights.append(f"📦 STOCK-OUT WEEKS: {impact:.1f}% LOWER (inventory issues)")

                # Autocorrelation (week-to-week consistency)
                if len(sales) > 2:
                    autocorr = np.corrcoef(sales[:-1], sales[1:])[0, 1]
                    if autocorr > 0.7:
                        insights.append(f"🔄 High week-to-week correlation ({autocorr:.2f}) - stable patterns")
                    elif autocorr < 0.3:
                        insights.append(f"🎲 Low week-to-week correlation ({autocorr:.2f}) - volatile/unpredictable")

                return "\n".join(insights) if insights else "No clear seasonal patterns detected"

            except Exception as e:
                return f"Error in seasonality analysis: {str(e)}"

        @tool
        def get_historical_data(num_weeks: str = "8") -> str:
            """
            Get detailed week-by-week sales with trend indicators.

            Args:
                num_weeks: Number of weeks to retrieve

            Returns:
                Formatted historical data with trends
            """
            try:
                import re
                numbers = re.findall(r'(\d+)', str(num_weeks).strip())
                num = int(numbers[0]) if numbers else 8

                if len(self.historical_data) == 0:
                    return "No historical data available."

                recent = self.historical_data.tail(num)
                sales_values = recent[self.sales_column].values

                result = f"📊 Last {len(recent)} weeks of sales:\n\n"

                for i, (idx, row) in enumerate(recent.iterrows()):
                    sales = row[self.sales_column]

                    # Add trend indicator
                    if i > 0:
                        prev_sales = sales_values[i-1]
                        change_pct = ((sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
                        trend = "↑" if change_pct > 5 else "↓" if change_pct < -5 else "→"
                        change_str = f"({change_pct:+.1f}%)"
                    else:
                        trend = "•"
                        change_str = ""

                    # Add flags
                    flags = []
                    if 'is_holiday' in row.index and row['is_holiday'] == 1:
                        flags.append("🎄HOLIDAY")
                    if 'is_post_holiday' in row.index and row['is_post_holiday'] == 1:
                        flags.append("⚠️POST-HOLIDAY")
                    if 'is_promotion' in row.index and row['is_promotion'] == 1:
                        flags.append("🎯PROMO")
                    if 'is_stockout' in row.index and row['is_stockout'] == 1:
                        flags.append("📦STOCKOUT")

                    flag_str = " ".join(flags)

                    result += f"  Week {i+1}: ${sales:>10,.2f} {trend} {change_str:>10} {flag_str}\n"

                # Add summary stats
                avg = np.mean(sales_values)
                result += f"\n  Average: ${avg:,.2f}"

                return result.strip()

            except Exception as e:
                return f"Error: {str(e)}"

        @tool
        def analyze_trends(query: str = "") -> str:
            """
            Advanced trend analysis with forecasting and pattern detection.

            Returns:
                Detailed trend analysis with momentum and acceleration
            """
            try:
                if len(self.historical_data) < 4:
                    return "Need at least 4 weeks for trend analysis."

                sales = self.historical_data[self.sales_column].values
                x = np.arange(len(sales))

                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, sales)

                mean_sales = np.mean(sales)
                pct_change_per_week = (slope / mean_sales * 100) if mean_sales > 0 else 0

                # Determine trend strength
                if abs(pct_change_per_week) < 1:
                    trend_desc = "STABLE"
                    emoji = "→"
                elif pct_change_per_week > 5:
                    trend_desc = "STRONGLY GROWING"
                    emoji = "📈"
                elif pct_change_per_week > 0:
                    trend_desc = "GROWING"
                    emoji = "↗️"
                elif pct_change_per_week < -5:
                    trend_desc = "STRONGLY DECLINING"
                    emoji = "📉"
                else:
                    trend_desc = "DECLINING"
                    emoji = "↘️"

                result = f"{emoji} Trend: {trend_desc}\n"
                result += f"Weekly change: ${slope:+,.2f} ({pct_change_per_week:+.2f}%)\n"
                result += f"Trend strength: R²={r_value**2:.3f} (p={p_value:.4f})"

                # Recent momentum (last 4 vs previous 4)
                if len(sales) >= 8:
                    recent_4 = np.mean(sales[-4:])
                    prev_4 = np.mean(sales[-8:-4])
                    momentum = ((recent_4 - prev_4) / prev_4 * 100) if prev_4 > 0 else 0

                    if abs(momentum) > 10:
                        result += f"\n🔥 Recent momentum: {momentum:+.1f}% (last 4 weeks vs prior 4)"

                # Acceleration (is trend speeding up or slowing down?)
                if len(sales) >= 12:
                    recent_slope = stats.linregress(np.arange(6), sales[-6:])[0]
                    older_slope = stats.linregress(np.arange(6), sales[-12:-6])[0]

                    if abs(recent_slope - slope) / abs(slope) > 0.5 if slope != 0 else abs(recent_slope) > mean_sales * 0.01:
                        if recent_slope > slope:
                            result += "\n⚡ Trend ACCELERATING (speeding up)"
                        else:
                            result += "\n🔻 Trend DECELERATING (slowing down)"

                return result

            except Exception as e:
                return f"Error: {str(e)}"

        @tool
        def detect_anomalies(query: str = "") -> str:
            """
            Detect unusual weeks that deviate from normal patterns.

            Returns:
                List of anomalous weeks and their characteristics
            """
            try:
                if len(self.historical_data) < 8:
                    return "Need at least 8 weeks to detect anomalies."

                sales = self.historical_data[self.sales_column].values
                mean = np.mean(sales)
                std = np.std(sales)

                anomalies = []

                for i, sale in enumerate(sales[-10:]):  # Check last 10 weeks
                    z_score = (sale - mean) / std if std > 0 else 0

                    if abs(z_score) > 2:  # More than 2 standard deviations
                        direction = "ABOVE" if z_score > 0 else "BELOW"
                        anomalies.append(f"Week {i+1}: ${sale:,.0f} ({direction} normal by {abs(z_score):.1f}σ)")

                if anomalies:
                    result = "⚠️ Detected anomalies in recent weeks:\n"
                    result += "\n".join(anomalies)
                    result += "\n\nConsider if these represent special events (holidays, promos, stock-outs)"
                    return result
                else:
                    return "✓ No significant anomalies detected in recent weeks"

            except Exception as e:
                return f"Error: {str(e)}"

        @tool
        def make_prediction(predicted_sales: str) -> str:
            """
            Record final sales prediction.

            Args:
                predicted_sales: Predicted amount (e.g., "50000")

            Returns:
                Confirmation with prediction details
            """
            try:
                import re
                sales_str = str(predicted_sales).strip().replace('$', '').replace(',', '')
                numbers = re.findall(r'([0-9]+\.?[0-9]*)', sales_str)

                if numbers:
                    sales_value = float(numbers[0])
                else:
                    sales_value = float(sales_str)

                # Calculate context (how this compares to historical average)
                if len(self.historical_data) > 0:
                    hist_avg = self.historical_data[self.sales_column].mean()
                    vs_avg = ((sales_value - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0

                    result = f"✓ Prediction recorded: ${sales_value:,.2f}\n"
                    result += f"Historical average: ${hist_avg:,.2f}\n"
                    result += f"Your prediction is {abs(vs_avg):.1f}% {'ABOVE' if vs_avg > 0 else 'BELOW'} average"
                else:
                    result = f"✓ Prediction recorded: ${sales_value:,.2f}"

                return result

            except Exception as e:
                return f"Error: {str(e)}"

        return [
            calculate_average_sales,
            check_seasonality,
            get_historical_data,
            analyze_trends,
            detect_anomalies,
            make_prediction
        ]
