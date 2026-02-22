# replit.md

## Overview

This is an AI-powered inventory management and supply chain forecasting web application. It reads supply chain data from a CSV file, uses Facebook Prophet for demand forecasting, calculates reorder points and stockout risks, and generates strategic AI insights using OpenAI's API. The app presents this information through a Flask web dashboard with multiple views: a command center homepage, demand forecasts, AI-generated strategic insights, and data visualizations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Structure
- **Backend**: Python Flask web application (`main.py` serves routes, `inventory_backend.py` contains core logic)
- **Frontend**: Server-side rendered HTML using Jinja2 templates with a shared base layout (`templates/base.html`)
- **Static Assets**: CSS styling in `static/css/style.css`, generated chart images stored in `static/charts/`

### Core Components

**InventoryForecast class** (`inventory_backend.py`):
- Loads and cleans CSV supply chain data using pandas
- Forecasts demand using Facebook Prophet (time-series forecasting library)
- Calculates reorder suggestions, safety stock levels, and stockout risk metrics
- Generates AI-powered strategic insights per product using OpenAI API (via Replit AI Integrations)
- Uses `ThreadPoolExecutor` for concurrent API calls and `tenacity` for retry logic with exponential backoff on rate limits
- Exports results and generates matplotlib chart images

**Flask Routes** (`main.py`):
- `/` — Home dashboard showing critical reorder alerts and stockout warnings with AI strategic advice
- `/forecasts` — Table view of predicted demand, inventory levels, and reorder suggestions per product
- `/insights` — Detailed AI-generated strategic insights per product with supplier and region context
- `/visualizations` — Grid of matplotlib-generated demand/inventory charts (PNG images)

### Data Flow
1. CSV file is loaded and cleaned (date parsing, sorting, missing value handling)
2. Prophet models are fit per product to predict next 4 weeks of demand
3. Reorder metrics (safety stock, stockout risk, reorder quantity) are calculated
4. OpenAI generates natural language strategic insights for each product
5. Results are cached in `backend.output_df` and served to templates
6. `refresh_data()` ensures data is processed before each route renders

### Design Decisions
- **Server-side rendering over SPA**: Chose Jinja2 templates for simplicity. The app is data-heavy but doesn't need real-time interactivity, making SSR appropriate.
- **In-memory data caching**: The `InventoryForecast` instance persists across requests, avoiding redundant CSV reads and API calls. The `refresh_data()` function checks if processing has already been done.
- **Prophet for forecasting**: Chosen for its ability to handle time-series data with seasonality out of the box, with a fallback to simple averaging when insufficient data exists.
- **Concurrent AI calls**: Uses `ThreadPoolExecutor` to parallelize OpenAI API calls across products, with retry logic for rate limiting.

### Frontend Architecture
- Single CSS file with a clean, card-based dashboard layout
- Color-coded alerts: yellow (`#ffc220`) for reorder warnings, red (`#e12026`) for stockout risks, green (`#008a3c`) for healthy status
- Responsive grid layouts for dashboard cards, insights, and chart displays
- Navigation bar with four main sections

### Data Source
- CSV file at `attached_assets/AISupplyChainProject(Sheet1)_1771129264461.csv`
- Expected columns include: Product, Date, Inventory, Category, Supplier, Region, Lead Time (days), On-time Delivery %
- Computed columns added during processing: Predicted_Weekly_Demand, Stockout_Risk, Reorder_Suggested, Safety_Stock, AI_Insight

## External Dependencies

### AI/ML Libraries
- **Facebook Prophet** (`prophet`): Time-series demand forecasting
- **OpenAI API** (via Replit AI Integrations): Generates strategic inventory insights using GPT models. Configured through `AI_INTEGRATIONS_OPENAI_API_KEY` and `AI_INTEGRATIONS_OPENAI_BASE_URL` environment variables. Uses model `gpt-5`.

### Python Packages
- **Flask**: Web framework for routing and template rendering
- **pandas**: Data manipulation and CSV processing
- **matplotlib**: Chart generation (saved as PNG files to `static/charts/`)
- **tenacity**: Retry logic with exponential backoff for API rate limit handling
- **concurrent.futures**: Parallel execution of OpenAI API calls

### Environment Variables Required
- `AI_INTEGRATIONS_OPENAI_API_KEY`: API key for OpenAI access through Replit's AI Integrations
- `AI_INTEGRATIONS_OPENAI_BASE_URL`: Base URL for the OpenAI-compatible API endpoint

### No Database
- Data is sourced entirely from a CSV file and processed in-memory using pandas DataFrames
- No database is used; if persistence or scaling is needed, a database layer would need to be added