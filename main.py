from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

from ml_models import SalesPredictor
from product_catalog import get_all_products, get_product_by_id, PRODUCT_CATALOG
from ai_insights import generate_product_insights

# Load environment variables
load_dotenv()

app = FastAPI(title="Advanced Sales Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML model
predictor = SalesPredictor()

# Try to load models at startup
try:
    model_loaded = predictor.load_models()
    if not model_loaded:
        print("Warning: Failed to load prediction models. Upload data to train models first.")
except Exception as e:
    print(f"Error loading models: {str(e)}")

def preprocess_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess historical data with advanced features"""
    try:
        # Create a copy to avoid modifying the original dataframe
        processed_df = df.copy()
        
        # Fill missing values first
        processed_df = processed_df.fillna({
            'sales': 0,
            'price': 0,
            'inventory': 0,
            'is_promotion': 'false',
            'temperature': 20,  # reasonable default temperature
            'competitor_price': 0,
            'stock_level': 0
        })
        
        # Convert boolean strings to actual booleans
        processed_df['is_promotion'] = processed_df['is_promotion'].map({'true': True, 'false': False}).fillna(False)
        
        # Basic preprocessing
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        processed_df['month'] = processed_df['date'].dt.month
        processed_df['day_of_week'] = processed_df['date'].dt.dayofweek
        processed_df['is_weekend'] = processed_df['day_of_week'].isin([5, 6]).astype(int)
        processed_df['is_holiday'] = 0  # You can add holiday detection logic
        
        # Ensure numeric columns are float
        numeric_columns = ['sales', 'price', 'inventory', 'temperature', 'competitor_price', 'stock_level']
        for col in numeric_columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0).astype(float)
        
        # Calculate rolling statistics per product
        processed_df['previous_sales'] = processed_df.groupby('product_id')['sales'].shift(1).fillna(0)
        processed_df['sales_ma7'] = processed_df.groupby('product_id')['sales'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        processed_df['sales_ma30'] = processed_df.groupby('product_id')['sales'].transform(lambda x: x.rolling(30, min_periods=1).mean())
        
        # Calculate promotion uplift
        processed_df['promotion_uplift'] = processed_df.apply(
            lambda row: row['sales'] / row['sales_ma7'] - 1 if row['is_promotion'] and row['sales_ma7'] > 0 else 0, 
            axis=1
        )
        
        # Convert boolean to int for is_promotion (after all calculations are done)
        processed_df['is_promotion'] = processed_df['is_promotion'].astype(int)
        
        # Prepare data for Prophet (ds = date, y = target variable)
        # Keep the original columns and add Prophet-specific ones
        processed_df['ds'] = processed_df['date']
        processed_df['y'] = processed_df['sales']
        
        return processed_df
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame head:\n{df.head()}")
        raise

@app.post("/upload-data")
async def upload_historical_data(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with open("temp_data.csv", "wb") as buffer:
            buffer.write(await file.read())
        
        # Load and preprocess data
        df = pd.read_csv("temp_data.csv")
        processed_df = preprocess_historical_data(df)
        
        # Train models
        predictor.train(processed_df)
        
        # Save models
        os.makedirs("models", exist_ok=True)
        predictor.save_models("models")
        
        # Cleanup
        os.remove("temp_data.csv")
        
        return {
            "message": "Models trained successfully",
            "data_points": len(df),
            "products": len(df['product_id'].unique())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products")
async def get_products():
    """Get all products with their categories"""
    return get_all_products()

@app.get("/products/{product_id}")
async def get_product(product_id: str):
    """Get detailed information about a specific product"""
    product = get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@app.post("/predict/{product_id}")
async def predict_sales(
    product_id: str, 
    days: int = 30,
    include_promotions: bool = True
):
    """Generate detailed sales predictions and insights for a product"""
    # Get product details
    product = get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product with ID {product_id} not found")
    
    try:
        # Check if models are loaded
        if not hasattr(predictor, 'prophet_model') or predictor.prophet_model is None:
            # Try to load models
            if not predictor.load_models():
                raise HTTPException(
                    status_code=400,
                    detail="Prediction models not trained. Please upload historical data first."
                )
        
        # Generate predictions
        predictions, insights = predictor.predict(product, days=days)
        
        # Add promotion scenarios if requested
        if include_promotions:
            # Create a copy of the product with promotion flag set
            promotion_product = product.copy()
            promotion_product['is_promotion'] = True
            
            # Generate predictions with promotion
            promotion_predictions, promotion_insights = predictor.predict(promotion_product, days=days)
            
            # Calculate promotion impact
            base_total_sales = sum(p['sales_forecast'] for p in predictions)
            promo_total_sales = sum(p['sales_forecast'] for p in promotion_predictions)
            sales_uplift = ((promo_total_sales - base_total_sales) / base_total_sales) * 100 if base_total_sales > 0 else 0
            
            # Determine optimal promotion timing
            sales_differences = []
            for i in range(len(predictions)):
                if i < len(promotion_predictions):
                    diff = promotion_predictions[i]['sales_forecast'] - predictions[i]['sales_forecast']
                    sales_differences.append((i, diff))
            
            # Find the best consecutive 7-day window for promotions
            best_window_start = 0
            best_window_sum = 0
            
            for start in range(len(sales_differences) - 7 + 1):
                window_sum = sum(diff for _, diff in sales_differences[start:start+7])
                if window_sum > best_window_sum:
                    best_window_sum = window_sum
                    best_window_start = start
            
            # Get the dates for the best window
            best_window_dates = [predictions[i]['date'] for i in range(best_window_start, best_window_start + 7)]
            
            # Format dates for display
            start_date = datetime.fromisoformat(best_window_dates[0].replace('Z', '+00:00')).strftime('%Y-%m-%d')
            end_date = datetime.fromisoformat(best_window_dates[-1].replace('Z', '+00:00')).strftime('%Y-%m-%d')
            
            # Create promotion insights
            promotion_insights = {
                "sales_uplift": sales_uplift,
                "recommended": sales_uplift > 5,  # Only recommend if uplift is significant
                "best_timing": f"{start_date} to {end_date}",
                "expected_uplift": sales_uplift,
                "window_start_idx": best_window_start,
                "window_end_idx": best_window_start + 6
            }
        
        # Calculate trend analysis
        if len(predictions) >= 2:
            first_week = sum(p['sales_forecast'] for p in predictions[:7]) / 7
            last_week = sum(p['sales_forecast'] for p in predictions[-7:]) / 7
            growth_rate = ((last_week - first_week) / first_week) * 100 if first_week > 0 else 0
            
            trend_analysis = {
                "direction": "up" if growth_rate > 0 else "down",
                "strength": abs(growth_rate) / 100,  # Normalize to 0-1 range
                "growth_rate": growth_rate
            }
        else:
            trend_analysis = {"direction": "stable", "strength": 0, "growth_rate": 0}
        
        # Generate market insights
        market_insights = {
            "seasonality_strength": np.random.uniform(0.1, 0.4),  # Mock data, would be calculated from real data
            "demand_stability": np.random.uniform(0.5, 0.9),  # Mock data
            "competition_level": np.random.choice(["low", "moderate", "high"]),  # Mock data
            "growth_potential": np.random.uniform(-5, 15)  # Mock data
        }
        
        # Calculate price optimization
        current_price = product['price']
        price_elasticity = -1.5  # Mock elasticity, would be calculated from real data
        
        # Simple price optimization model
        price_points = [current_price * (1 + (i - 5) / 100) for i in range(11)]  # -5% to +5%
        expected_sales = []
        expected_revenue = []
        
        for price in price_points:
            # Apply price elasticity model
            price_change_pct = (price - current_price) / current_price
            sales_change_pct = price_change_pct * price_elasticity
            
            base_sales = sum(p['sales_forecast'] for p in predictions[:7]) / 7
            adjusted_sales = base_sales * (1 + sales_change_pct)
            
            expected_sales.append(adjusted_sales)
            expected_revenue.append(adjusted_sales * price)
        
        # Find optimal price for revenue
        max_revenue_idx = expected_revenue.index(max(expected_revenue))
        optimal_price = price_points[max_revenue_idx]
        
        price_optimization = {
            "optimal_price": optimal_price,
            "expected_revenue_increase": ((max(expected_revenue) - (base_sales * current_price)) / (base_sales * current_price)) * 100,
            "price_points": price_points,
            "expected_sales": expected_sales,
            "expected_revenue": expected_revenue
        }
        
        # Combine all insights
        insights = {
            "trend_analysis": trend_analysis,
            "market_insights": market_insights,
            "price_optimization": price_optimization
        }
        
        if include_promotions:
            insights["promotion_impact"] = promotion_insights
        
        # Generate natural language insights
        nl_insights = generate_nl_insights(product, insights, predictions)
        
        # Generate enhanced AI-powered insights with product catalog for related products
        from ai_insights import generate_product_insights
        from product_catalog import PRODUCT_CATALOG
        
        ai_insights = generate_product_insights(product, predictions, insights, PRODUCT_CATALOG)
        
        return {
            "product": product,
            "predictions": predictions,
            "insights": insights,
            "natural_language_insights": nl_insights,
            "ai_insights": ai_insights
        }
    
        raise
    except Exception as e:
        print(f"Unexpected error in predict_sales: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_nl_insights(product: Dict, insights: Dict, predictions: List[Dict]) -> Dict:
    """Generate natural language insights from the ML predictions"""
    
    # Price optimization insights
    base_price = product.get('base_price', product.get('price', 0))
    price_change = insights['price_optimization']['optimal_price'] - base_price
    revenue_increase = insights['price_optimization'].get('expected_revenue_increase', 0)
    price_message = (
        f"Consider {'increasing' if price_change > 0 else 'decreasing'} the price "
        f"by ₹{abs(price_change):.2f} to optimize revenue. This could lead to "
        f"a {abs(revenue_increase):.1f}% improvement."
    )
    
    # Sales trend insights
    trend = insights['trend_analysis']
    trend_message = (
        f"Sales are showing a {trend['direction']} trend with "
        f"{trend['strength']*100:.1f}% {'growth' if trend['direction'] == 'up' else 'decline'} "
        f"rate."
    )
    
    # Promotion recommendations
    if 'promotion_impact' in insights:
        promo = insights['promotion_impact']
        promo_message = (
            f"{'Recommended' if promo.get('recommended', False) else 'Not recommended'} to run promotions. "
            f"Expected sales uplift from promotions: {promo.get('expected_uplift', 0):.1f}%. "
            f"Best timing: {promo.get('best_timing', 'N/A')}."
        )
    else:
        promo_message = "Promotion analysis not available."
    
    # Market analysis
    market = insights['market_insights']
    market_message = (
        f"Product shows {'strong' if market.get('seasonality_strength', 0) > 0.2 else 'weak'} "
        f"seasonal patterns. Demand stability is "
        f"{'high' if market.get('demand_stability', 0) > 0.7 else 'moderate' if market.get('demand_stability', 0) > 0.4 else 'low'}. "
        f"Growth potential: {market.get('growth_potential', 0):.1f}%."
    )
    
    # Short-term forecast
    next_week = sum(p.get('sales_forecast', 0) for p in predictions[:min(7, len(predictions))]) / min(7, len(predictions)) if predictions else 0
    next_month = sum(p.get('sales_forecast', 0) for p in predictions[:min(30, len(predictions))]) / min(30, len(predictions)) if predictions else 0
    forecast_message = (
        f"Expected average daily sales: {next_week:.1f} units next week, "
        f"{next_month:.1f} units over the next month."
    )
    
    return {
        "summary": f"Key insights for {product['name']}:",
        "price_optimization": price_message,
        "sales_trend": trend_message,
        "promotion_strategy": promo_message,
        "market_analysis": market_message,
        "short_term_forecast": forecast_message
    }

# This function is now deprecated - we use the enhanced generate_product_insights from ai_insights.py
def generate_ai_insights(product: Dict, predictions: List[Dict], insights: Dict) -> Dict:
    """Generate AI-powered insights from the ML predictions"""
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('gemini-pro')
        
        # Calculate additional metrics
        total_sales = sum(p['sales_forecast'] for p in predictions)
        avg_sales = total_sales / len(predictions)
        price_change = insights['price_optimization']['optimal_price'] - product.get('price', 0)
        revenue_increase = insights['price_optimization']['expected_revenue_increase']
        
        # Create a detailed prompt with all available data
        prompt = f"""
        Generate a comprehensive market analysis for {product['name']} based on the following data:

        Product Information:
        - Name: {product['name']}
        - Category: {product['category']}
        - Current Price: ${product.get('price', 0):.2f}
        - Inventory Level: {product.get('inventory', 0)} units

        Sales Metrics:
        - Average Daily Sales: {avg_sales:.1f} units
        - Total Forecast Sales: {total_sales:.1f} units
        - Sales Trend: {insights['trend_analysis']['direction']} ({insights['trend_analysis']['strength']*100:.1f}% change)
        - Demand Stability: {insights['market_insights']['demand_stability']*100:.1f}%

        Market Position:
        - Market Share: {insights['market_insights'].get('market_share', 0)}%
        - Competition Level: {insights['market_insights'].get('competition_level', 'Moderate')}
        - Optimal Price Point: ${insights['price_optimization']['optimal_price']:.2f}
        - Expected Revenue Increase: {revenue_increase:.1f}%

        Provide a detailed analysis in these areas:

        1. Market Position & Strategy:
        - Current market positioning
        - Competitive advantages/disadvantages
        - Strategic recommendations

        2. Sales Performance Analysis:
        - Sales trend analysis
        - Performance vs. market
        - Key growth drivers/barriers

        3. Customer Recommendations:
        - Target customer segments
        - Purchase patterns
        - Value proposition

        4. Future Market Outlook:
        - Short-term forecast (30 days)
        - Growth opportunities
        - Risk factors
        - Action items

        Keep each section concise but data-driven and actionable.
        """
        
        response = model.generate_content(prompt)
        ai_text = response.text
        
        # Split the AI response into sections
        sections = ai_text.split('\n\n')
        
        # Extract sections using more robust parsing
        def extract_section(text: str, section_name: str) -> str:
            for section in sections:
                if section_name.lower() in section.lower():
                    return section.split('\n', 1)[1] if '\n' in section else section
            return f"Analysis for {section_name} not available"
        
        market_position = extract_section(ai_text, "Market Position & Strategy")
        sales_performance = extract_section(ai_text, "Sales Performance Analysis")
        customer_recommendations = extract_section(ai_text, "Customer Recommendations")
        future_outlook = extract_section(ai_text, "Future Market Outlook")
        
        # Add competitor analysis if available
        competitor_analysis = {
            "competitors": [
                {
                    "name": "Competitor 1",
                    "price": product.get('price', 0) * 1.1
                },
                {
                    "name": "Competitor 2",
                    "price": product.get('price', 0) * 0.9
                }
            ],
            "market_stats": {
                "average_price": product.get('price', 0) * 1.05,
                "min_price": product.get('price', 0) * 0.8,
                "max_price": product.get('price', 0) * 1.2,
                "price_range": product.get('price', 0) * 0.4,
                "our_position": "competitive",
                "price_difference_percent": 5.0
            }
        }
        
        # Add sentiment analysis
        sentiment_analysis = {
            "sentiment_score": 0.75,
            "review_count": 150,
            "positive_themes": ["quality", "value", "reliability"],
            "negative_themes": ["packaging"],
            "recommendation_rate": 85
        }
        
    except Exception as e:
        print(f"Error generating Gemini insights: {str(e)}")
        return {
            "detailed_analysis": {
                "market_position": "AI insights currently unavailable. Please check your API key configuration.",
                "sales_performance": "",
                "customer_recommendations": "",
                "future_outlook": ""
            },
            "success": False
        }
    
    return {
        "product_definition": f"Detailed analysis for {product['name']} in {product['category']} category",
        "competitor_analysis": competitor_analysis,
        "sentiment_analysis": sentiment_analysis,
        "detailed_analysis": {
            "market_position": market_position,
            "sales_performance": sales_performance,
            "customer_recommendations": customer_recommendations,
            "future_outlook": future_outlook
        },
        "success": True
    }

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8002)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
