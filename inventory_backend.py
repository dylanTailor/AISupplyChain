import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from openai import OpenAI
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

AI_INTEGRATIONS_OPENAI_API_KEY = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
AI_INTEGRATIONS_OPENAI_BASE_URL = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")

# Only initialize client if API key is present
if AI_INTEGRATIONS_OPENAI_API_KEY:
    openai = OpenAI(
        api_key=AI_INTEGRATIONS_OPENAI_API_KEY,
        base_url=AI_INTEGRATIONS_OPENAI_BASE_URL
    )
else:
    openai = None

def is_rate_limit_error(exception: BaseException) -> bool:
    """Check if the exception is a rate limit error."""
    return "429" in str(exception) or "RATELIMIT_EXCEEDED" in str(exception)

class InventoryForecast:
    """
    Core logic for reading supply chain data, forecasting demand with Prophet,
    calculating reorder metrics, and generating AI-powered insights.
    """
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.output_df = None

    def load_and_clean_data(self):
        """Read CSV, convert dates, and handle missing values."""
        self.df = pd.read_csv(self.csv_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        # Sort by product and date for time-series analysis
        self.df = self.df.sort_values(['Product', 'Date'])
        self.df = self.df.fillna(0)
        return self.df

    def forecast_demand(self):
        """
        Predict next 4 weeks of demand using Facebook Prophet.
        Falls back to simple averaging if insufficient data exists.
        """
        products = self.df['Product'].unique()
        self.df['Predicted_Weekly_Demand'] = 0.0
        
        for product in products:
            prod_df = self.df[self.df['Product'] == product].copy()
            
            # Prophet requires at least 2 data points
            if len(prod_df.dropna(subset=['Weekly Demand'])) < 2:
                avg_demand = prod_df['Weekly Demand'].mean()
                latest_idx = prod_df.index[-1]
                self.df.at[latest_idx, 'Predicted_Weekly_Demand'] = max(0, avg_demand)
                continue

            # Prepare data for Prophet ('ds' and 'y')
            prophet_df = prod_df[['Date', 'Weekly Demand']].rename(columns={'Date': 'ds', 'Weekly Demand': 'y'})
            
            model = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=4, freq='W')
            forecast = model.predict(future)
            
            # Forecast result is average of the next 4 predicted weeks
            predicted_val = forecast.tail(4)['yhat'].mean()
            
            latest_idx = prod_df.index[-1]
            self.df.at[latest_idx, 'Predicted_Weekly_Demand'] = max(0, float(predicted_val))

        # Calculate metrics for current snapshot
        self.df['Stockout_Risk'] = self.df['Inventory'] - self.df['Predicted_Weekly_Demand']
        self.df['Reorder_Suggested'] = (self.df['Predicted_Weekly_Demand'] + self.df['Safety Stock'] - self.df['Inventory']).clip(lower=0)
        
        return self.df

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_rate_limit_error),
        reraise=True
    )
    def get_ai_insight(self, row_data):
        """Call OpenAI for a natural language analysis and safety stock recommendation."""
        if not AI_INTEGRATIONS_OPENAI_API_KEY or not openai:
            return {
                "insight": f"Current inventory is {row_data['Inventory']} units with a predicted demand of {row_data['Predicted_Weekly_Demand']:.1f}. Recommendation: Maintain safety stock levels.",
                "safety_stock": row_data.get('Safety Stock', 50)
            }

        prompt = f"""
        Analyze this inventory data for product: {row_data['Product']}
        Current Inventory: {row_data['Inventory']}
        Predicted Weekly Demand: {row_data['Predicted_Weekly_Demand']:.2f}
        Stockout Risk: {row_data['Stockout_Risk']:.2f}
        Safety Stock: {row_data['Safety Stock']}
        Lead Time: {row_data['Lead Time (days)']} days
        Supplier: {row_data['Supplier']}
        On-time Delivery: {row_data['On-time Delivery %']}%

        Provide a JSON response with:
        1. "insight": A concise, friendly analysis (2 sentences max). DO NOT start with "Strategic analysis for...". Just the insight.
        2. "recommended_safety_stock": A numerical value for the ideal safety stock level based on the risk.
        """
        
        response = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            max_completion_tokens=250
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "insight": data.get("insight", "").strip(),
            "safety_stock": data.get("recommended_safety_stock", row_data.get('Safety Stock', 50))
        }

    def generate_all_insights(self):
        """Generate AI insights and update safety stock for the latest snapshot of each product."""
        latest_data = self.df.groupby('Product').last().reset_index()
        latest_data['AI_Insight'] = ""
        
        for idx, row in latest_data.iterrows():
            try:
                result = self.get_ai_insight(row)
                latest_data.at[idx, 'AI_Insight'] = result["insight"]
                # Update safety stock in the main dataframe for this product's latest entry
                product_mask = (self.df['Product'] == row['Product']) & (self.df['Date'] == row['Date'])
                self.df.loc[product_mask, 'Safety Stock'] = result["safety_stock"]
                latest_data.at[idx, 'Safety Stock'] = result["safety_stock"]
            except Exception as e:
                latest_data.at[idx, 'AI_Insight'] = f"Insight unavailable: {str(e)}"
            
        self.output_df = latest_data
        return latest_data

    def export_results(self, output_path='ai_inventory_output.xlsx'):
        """Save results to Excel with enhanced formatting and Power BI calculated columns."""
        if self.output_df is not None:
            export_df = self.output_df.copy()
            
            # 1. Reorder_Suggested (already calculated in forecast_demand, but ensure it matches logic)
            if 'Safety Stock' in export_df.columns:
                export_df['Reorder_Suggested'] = (export_df['Predicted_Weekly_Demand'] + export_df['Safety Stock'] - export_df['Inventory']).clip(lower=0)
            
            # 2. Stock_Status
            def get_stock_status(row):
                if row['Inventory'] < row['Predicted_Weekly_Demand']:
                    return "Understocked"
                elif row['Inventory'] > row['Predicted_Weekly_Demand']:
                    return "Overstocked"
                else:
                    return "Balanced"
            export_df['Stock_Status'] = export_df.apply(get_stock_status, axis=1)
            
            # 3. Inventory_Value & 4. Potential_Revenue
            # Handle cases where columns might be missing with defaults
            unit_cost_col = 'Unit Cost' if 'Unit Cost' in export_df.columns else 'Unit Price' # Fallback check
            selling_price_col = 'Selling Price' if 'Selling Price' in export_df.columns else 'Price'
            
            if unit_cost_col in export_df.columns:
                export_df['Inventory_Value'] = export_df['Inventory'] * export_df[unit_cost_col]
            else:
                export_df['Inventory_Value'] = 0.0 # Default if missing
                
            if selling_price_col in export_df.columns:
                export_df['Potential_Revenue'] = export_df['Predicted_Weekly_Demand'] * export_df[selling_price_col]
            else:
                export_df['Potential_Revenue'] = 0.0 # Default if missing
            
            # 5. Risk_Level
            def get_risk_level(row):
                if row['Stockout_Risk'] < 0:
                    return "High Risk"
                elif 0 <= row['Stockout_Risk'] <= 20:
                    return "Medium Risk"
                else:
                    return "Low Risk"
            export_df['Risk_Level'] = export_df.apply(get_risk_level, axis=1)
            
            # 6. Supplier_Risk_Flag
            if 'On-time Delivery %' in export_df.columns:
                export_df['Supplier_Risk_Flag'] = export_df['On-time Delivery %'].apply(
                    lambda x: "Supplier Risk" if x < 85 else "Reliable"
                )
            
            # Final Column Selection and Ordering for Power BI
            cols_order = [
                'Product', 'Date', 'Inventory', 'Weekly Demand', 'Predicted_Weekly_Demand', 
                'Stockout_Risk', 'Safety Stock', 'Lead Time (days)', 'Supplier', 
                'Reorder_Suggested', 'Stock_Status', 'Inventory_Value', 
                'Potential_Revenue', 'Risk_Level', 'Supplier_Risk_Flag', 'AI_Insight'
            ]
            
            # Filter to columns that actually exist in the dataframe
            final_cols = [c for c in cols_order if c in export_df.columns]
            export_df = export_df[final_cols]
            
            # Format numbers
            numeric_cols = ['Predicted_Weekly_Demand', 'Stockout_Risk', 'Inventory_Value', 'Potential_Revenue', 'Reorder_Suggested']
            for col in numeric_cols:
                if col in export_df.columns:
                    export_df[col] = export_df[col].round(2)
            
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Inventory Analysis')
                
                workbook = writer.book
                worksheet = writer.sheets['Inventory Analysis']
                
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D7E4BC',
                    'border': 1
                })
                
                for col_num, value in enumerate(export_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                for i, col in enumerate(export_df.columns):
                    column_len = max(export_df[col].astype(str).str.len().max(), len(col)) + 2
                    worksheet.set_column(i, i, min(column_len, 50))


    def create_plot(self, product_name, output_dir='static/charts'):
        """Generate static charts for the frontend."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        prod_data = self.df[self.df['Product'] == product_name]
        plt.figure(figsize=(10, 5))
        plt.plot(prod_data['Date'], prod_data['Weekly Demand'], label='Historical Demand', marker='o')
        plt.axhline(y=prod_data['Inventory'].iloc[-1], color='r', linestyle='--', label='Current Inventory')
        
        # Add a point for the prediction
        latest_date = prod_data['Date'].iloc[-1]
        predicted_demand = prod_data['Predicted_Weekly_Demand'].iloc[-1]
        plt.scatter(latest_date, predicted_demand, color='green', s=100, label='AI Forecast', zorder=5)
        
        plt.title(f'Demand Forecast: {product_name}')
        plt.xlabel('Date')
        plt.ylabel('Units')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{product_name.replace(" ", "_")}.png')
        plt.close()
