import google.generativeai as genai
import os
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-pro')  # Updated model name to latest version

def analyze_price_trend(product: Dict, market_data: Dict) -> str:
    """Analyze price trends and market position"""
    current_price = product['price']
    competitor_price = market_data.get('competitor_price', current_price)
    optimal_price = market_data['price_optimization']['optimal_price']
    
    price_diff_pct = ((current_price - competitor_price) / competitor_price) * 100
    position = "higher" if price_diff_pct > 0 else "lower"
    
    return f"""
    Current Market Position:
    - Our price: ₹{current_price:.2f}
    - Market average: ₹{competitor_price:.2f}
    - Price difference: {abs(price_diff_pct):.1f}% {position} than market average
    - Optimal price point: ₹{optimal_price:.2f}
    """

def analyze_sales_performance(predictions: List[Dict], market_data: Dict) -> str:
    """Analyze sales performance and trends"""
    avg_daily_sales = sum(p['sales_forecast'] for p in predictions[:30]) / 30
    peak_sales = max(p['sales_forecast'] for p in predictions[:30])
    peak_day = max(predictions[:30], key=lambda x: x['sales_forecast'])['date']
    
    trend_strength = market_data['trend_analysis']['strength']
    trend_direction = market_data['trend_analysis']['direction']
    
    return f"""
    Sales Performance Analysis:
    - Average daily sales forecast: {avg_daily_sales:.1f} units
    - Peak sales expected on: {peak_day} ({peak_sales:.1f} units)
    - Overall trend: {abs(trend_strength*100):.1f}% {trend_direction}
    - Market stability: {market_data['market_insights']['demand_stability']*100:.1f}% stable
    """

def generate_buying_guide(product: Dict, market_data: Dict, predictions: List[Dict]) -> str:
    """Generate a buying guide for customers"""
    seasonality = "high" if market_data['market_insights']['seasonality_strength'] > 0.3 else "moderate"
    best_time = "during promotions" if market_data['promotion_impact']['recommended'] else "at current price"
    
    return f"""
    Buying Guide:
    - Best time to buy: {best_time}
    - Seasonal impact: {seasonality} seasonality observed
    - Expected savings during promotions: {market_data['promotion_impact']['expected_uplift']:.1f}%
    - Product stability: {market_data['market_insights']['demand_stability']*100:.1f}% stable demand
    """

def generate_product_definition(product: Dict) -> str:
    """Generate a detailed product definition using AI"""
    try:
        prompt = f"""
        Generate a detailed, informative, and engaging product definition for:
        
        Product: {product['name']}
        Category: {product['category']}
        Subcategory: {product['subcategory']}
        
        Include:
        1. A brief overview of what this product is
        2. Key features and benefits
        3. Common uses
        4. Any relevant nutritional/health information (if applicable)
        5. Storage recommendations (if applicable)
        
        Keep it concise (100-150 words) but comprehensive and customer-focused.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating product definition: {str(e)}")
        return f"Description for {product['name']} currently unavailable."

def analyze_competitor_products(product: Dict) -> Dict:
    """Analyze competitor products and pricing"""
    # This would ideally use a real market API
    # For demo purposes, we're generating simulated data
    
    base_price = product['price']
    category = product['category']
    subcategory = product['subcategory']
    
    # Simulate competitor data
    competitors = [
        {"name": "CompetitorA", "price": round(base_price * (0.85 + random.random() * 0.3), 2)},
        {"name": "CompetitorB", "price": round(base_price * (0.9 + random.random() * 0.2), 2)},
        {"name": "CompetitorC", "price": round(base_price * (0.95 + random.random() * 0.15), 2)}
    ]
    
    # Calculate market statistics
    competitor_prices = [c["price"] for c in competitors]
    avg_price = sum(competitor_prices) / len(competitor_prices) if competitor_prices else base_price
    min_price = min(competitor_prices) if competitor_prices else base_price * 0.9
    max_price = max(competitor_prices) if competitor_prices else base_price * 1.1
    
    # Determine if our price is competitive
    price_position = "higher" if base_price > avg_price else "lower" if base_price < avg_price else "at market average"
    price_diff = abs(((base_price - avg_price) / avg_price) * 100)
    
    return {
        "competitors": competitors,
        "market_stats": {
            "average_price": avg_price,
            "min_price": min_price,
            "max_price": max_price,
            "price_range": max_price - min_price,
            "our_position": price_position,
            "price_difference_percent": price_diff
        }
    }

def analyze_customer_sentiment(product: Dict) -> Dict:
    """Analyze customer sentiment and feedback"""
    # This would ideally use real customer review data
    # For demo purposes, we're generating simulated data
    
    # Generate random sentiment scores (would be real in production)
    sentiment_score = random.uniform(3.5, 4.8)  # Scale of 1-5
    review_count = random.randint(10, 200)
    
    # Generate key themes from "reviews"
    positive_themes = [
        "Quality", "Value for money", "Durability", "Effectiveness"
    ]
    negative_themes = [
        "Packaging", "Delivery time", "Size accuracy"
    ]
    
    # Select a random subset of themes
    selected_positive = random.sample(positive_themes, k=min(2, len(positive_themes)))
    selected_negative = random.sample(negative_themes, k=min(1, len(negative_themes)))
    
    return {
        "sentiment_score": sentiment_score,
        "review_count": review_count,
        "positive_themes": selected_positive,
        "negative_themes": selected_negative,
        "recommendation_rate": round(random.uniform(0.7, 0.95) * 100, 1)  # % who recommend
    }

def generate_related_products(product: Dict, catalog: Dict) -> List[Dict]:
    """Generate related product recommendations"""
    related_products = []
    
    try:
        # Find products in the same category
        category = product['category']
        subcategory = product['subcategory']
        product_id = product['id']
        
        # Get products from the same subcategory
        if category in catalog and subcategory in catalog[category]:
            for item in catalog[category][subcategory]:
                if item['id'] != product_id:  # Don't include the current product
                    related_products.append(item)
        
        # If we don't have enough, get products from other subcategories
        if len(related_products) < 3:
            for sub, items in catalog[category].items():
                if sub != subcategory:
                    for item in items:
                        related_products.append(item)
                        if len(related_products) >= 3:
                            break
                if len(related_products) >= 3:
                    break
        
        # Limit to top 3 related products
        return related_products[:3]
    except Exception as e:
        print(f"Error generating related products: {str(e)}")
        return []

def generate_product_insights(product: Dict, predictions: List[Dict], market_data: Dict, catalog: Dict = None) -> Dict:
    """Generate comprehensive natural language insights about the product using Gemini"""
    
    # Prepare detailed analysis
    price_analysis = analyze_price_trend(product, market_data)
    sales_analysis = analyze_sales_performance(predictions, market_data)
    buying_guide = generate_buying_guide(product, market_data, predictions)
    
    # Generate additional insights
    product_definition = generate_product_definition(product)
    competitor_analysis = analyze_competitor_products(product)
    sentiment_analysis = analyze_customer_sentiment(product)
    related_products = generate_related_products(product, catalog) if catalog else []
    
    # Create a detailed prompt for Gemini
    prompt = f"""
    As a retail analytics expert, provide detailed insights about this product:
    
    Product Information:
    - Name: {product['name']}
    - Category: {product['category']}
    - Subcategory: {product['subcategory']}
    - Current Price: ₹{product['price']}
    
    Product Definition:
    {product_definition}
    
    {price_analysis}
    
    {sales_analysis}
    
    {buying_guide}
    
    Competitor Analysis:
    - Average market price: ₹{competitor_analysis['market_stats']['average_price']:.2f}
    - Our price is {competitor_analysis['market_stats']['price_difference_percent']:.1f}% {competitor_analysis['market_stats']['our_position']} than market average
    - Price range in market: ₹{competitor_analysis['market_stats']['min_price']:.2f} to ₹{competitor_analysis['market_stats']['max_price']:.2f}
    
    Customer Sentiment:
    - Average rating: {sentiment_analysis['sentiment_score']:.1f}/5.0 ({sentiment_analysis['review_count']} reviews)
    - {sentiment_analysis['recommendation_rate']}% of customers recommend this product
    - Positive themes: {', '.join(sentiment_analysis['positive_themes'])}
    - Areas for improvement: {', '.join(sentiment_analysis['negative_themes'])}
    
    Please provide a comprehensive analysis covering:
    1. Market Position & Competitiveness:
       - How does our price compare to the market?
       - What is our competitive advantage?
       - Should we adjust our pricing strategy?
    
    2. Sales Performance & Trends:
       - What are the key sales patterns?
       - Which factors are driving sales?
       - What are the growth opportunities?
    
    3. Customer Recommendations:
       - When should customers buy?
       - What are the best deals to look for?
       - How can customers maximize value?
    
    4. Future Outlook:
       - What are the expected market changes?
       - How should we adapt our strategy?
       - What risks and opportunities exist?
    
    Format the response in clear sections with bullet points. Keep it business-focused and actionable.
    Use Indian Rupees (₹) for all price values.
    """
    
    try:
        response = model.generate_content(prompt)
        insights = response.text
        
        # Split insights into sections for better frontend display
        sections = insights.split('\n\n')
        
        return {
            "detailed_analysis": {
                "market_position": sections[0] if len(sections) > 0 else "",
                "sales_performance": sections[1] if len(sections) > 1 else "",
                "customer_recommendations": sections[2] if len(sections) > 2 else "",
                "future_outlook": sections[3] if len(sections) > 3 else ""
            },
            "product_definition": product_definition,
            "price_analysis": price_analysis,
            "sales_analysis": sales_analysis,
            "buying_guide": buying_guide,
            "competitor_analysis": competitor_analysis,
            "sentiment_analysis": sentiment_analysis,
            "related_products": related_products,
            "raw_insights": insights,
            "success": True
        }
    except Exception as e:
        print(f"Error generating Gemini insights: {str(e)}")
        return {
            "detailed_analysis": {
                "market_position": "Unable to generate market position analysis.",
                "sales_performance": "Unable to generate sales performance analysis.",
                "customer_recommendations": "Unable to generate customer recommendations.",
                "future_outlook": "Unable to generate future outlook."
            },
            "product_definition": f"Description for {product['name']} currently unavailable.",
            "competitor_analysis": {"competitors": [], "market_stats": {}},
            "sentiment_analysis": {},
            "related_products": [],
            "success": False
        }
