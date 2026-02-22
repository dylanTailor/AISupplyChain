from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from inventory_backend import InventoryForecast

app = Flask(__name__)

# CSV path based on attached assets
CSV_PATH = os.path.join(os.path.dirname(__file__), "attached_assets/AISupplyChainProject(Sheet1)_1771129264461.csv")
backend = InventoryForecast(CSV_PATH)
GLOBAL_SAFETY_STOCK = 50

def refresh_data():
    """Helper to ensure backend data is processed."""
    backend.load_and_clean_data()
    # Initial pass with global safety stock
    if 'Safety Stock' in backend.df.columns:
        backend.df['Safety Stock'] = GLOBAL_SAFETY_STOCK
    backend.forecast_demand()
    
    if backend.output_df is None:
        # This will now update safety stock based on AI recommendations
        backend.generate_all_insights()
        # Re-run demand forecast to update Reorder_Suggested and Stockout_Risk with NEW safety stock
        backend.forecast_demand()
        # Regenerate output_df with the updated metrics
        backend.output_df = backend.df.groupby('Product').last().reset_index()
        # Re-apply insights to the fresh output_df
        backend.generate_all_insights() 
        backend.export_results()

@app.route('/')
def index():
    refresh_data()
    results = backend.output_df
    
    # Identify critical actions for the Home page
    reorder_alerts = results[results['Reorder_Suggested'] > 0].to_dict('records')
    stockout_alerts = results[results['Stockout_Risk'] < 0].to_dict('records')
    
    return render_template('index.html', reorder_alerts=reorder_alerts, stockout_alerts=stockout_alerts, current_safety_stock=GLOBAL_SAFETY_STOCK)

@app.route('/update_safety_stock', methods=['POST'])
def update_safety_stock():
    global GLOBAL_SAFETY_STOCK
    try:
        GLOBAL_SAFETY_STOCK = int(request.form.get('safety_stock', 50))
        # Reset output_df to force re-generation of insights with new safety stock
        backend.output_df = None
    except ValueError:
        pass
    return redirect(url_for('index'))

@app.route('/forecasts')
def forecasts():
    refresh_data()
    products_data = backend.df.groupby('Product').last().reset_index().to_dict('records')
    return render_template('forecasts.html', products=products_data)

@app.route('/insights')
def insights():
    refresh_data()
    results = backend.output_df.to_dict('records')
    return render_template('insights.html', results=results)

@app.route('/visualizations')
def visualizations():
    refresh_data()
    products = backend.df['Product'].unique()
    # Generate/ensure charts exist for visual grid
    charts = []
    for prod in products[:8]: # Show top 8 for UI balance
        filename = f"{prod.replace(' ', '_')}.png"
        try:
            backend.create_plot(prod)
            charts.append(filename)
        except Exception:
            # If plot fails, don't add to list to avoid blank images
            pass
    
    return render_template('visualizations.html', charts=charts)

@app.route('/download_report')
def download_report():
    refresh_data()
    backend.export_results()
    from flask import send_file
    return send_file('../ai_inventory_output.xlsx', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
