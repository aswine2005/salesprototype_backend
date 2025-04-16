import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from datetime import datetime, timedelta
import joblib
import json
import os
from typing import Dict, List, Tuple, Union

# Type for numeric values that need conversion
NumericType = Union[np.integer, np.floating, np.ndarray, np.bool_, float, int, bool]

class SalesPredictor:
    def __init__(self):
        self.prophet_model = None
        self.price_impact_model = None
        self.promotion_impact_model = None
        self.scaler = StandardScaler()
        
    def to_native(self, value: NumericType) -> Union[float, bool, list]:
        """Convert numpy types to Python native types"""
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value
        
    def train(self, historical_data: pd.DataFrame):
        """Train all models using historical data"""
        # Group data by product_id for training
        product_groups = historical_data.groupby('product_id')
        
        # Train Prophet model for base sales prediction
        # Use the first product's data for initial training
        first_product_data = next(iter(product_groups))[1]
        
        # Prepare data for Prophet (time series)
        prophet_data = first_product_data[['date', 'y']].rename(columns={'date': 'ds', 'y': 'y'})
        
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        self.prophet_model.add_regressor('price')
        self.prophet_model.add_regressor('is_promotion')
        
        # Select only required columns for Prophet
        prophet_data = first_product_data[['ds', 'y', 'price', 'is_promotion']]
        self.prophet_model.fit(prophet_data)
        
        # Prepare features for price impact model
        X_price = historical_data[[
            'price', 'month', 'day_of_week', 'is_holiday', 
            'temperature', 'is_promotion', 'competitor_price'
        ]]
        y_price = historical_data['y']  # Using 'y' instead of 'sales'
        
        # Scale features
        X_price_scaled = self.scaler.fit_transform(X_price)
        
        # Train GradientBoosting for price impact
        self.price_impact_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.price_impact_model.fit(X_price_scaled, y_price)
        
        # Train promotion impact model
        X_promo = historical_data[[
            'price', 'month', 'day_of_week', 'is_holiday',
            'previous_sales', 'stock_level'
        ]]
        y_promo = historical_data['promotion_uplift']
        
        self.promotion_impact_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            random_state=42
        )
        self.promotion_impact_model.fit(X_promo, y_promo)

    def predict(self, 
                product_data: Dict,
                days: int = 30,
                include_promotions: bool = True) -> Tuple[List[Dict], Dict]:
        """Generate predictions and insights"""
        
        # Generate future dates
        future_dates = pd.date_range(start=datetime.now(), periods=days)
        
        # Create future dataframe for Prophet
        future_df = pd.DataFrame({
            'ds': future_dates,
            'price': [product_data['price']] * days,
            'is_promotion': [0] * days  # Will be updated for promotion scenarios
        })
        
        # Get base forecast
        forecast = self.prophet_model.predict(future_df)
        
        # Generate price sensitivity analysis
        price_variations = np.linspace(0.8, 1.2, 5) * product_data['price']
        price_impacts = []
        
        for price in price_variations:
            features = np.array([[
                price,
                future_dates[0].month,
                future_dates[0].weekday(),
                0,  # is_holiday
                25,  # temperature (example value)
                0,  # is_promotion
                product_data['price']  # competitor_price
            ]])
            scaled_features = self.scaler.transform(features)
            impact = self.price_impact_model.predict(scaled_features)[0]
            price_impacts.append({'price': price, 'impact': impact})
        
        # Find optimal price
        optimal_price = max(price_impacts, key=lambda x: x['impact'])['price']
        
        # Generate promotion recommendations
        promo_features = np.array([[
            product_data['price'],
            future_dates[0].month,
            future_dates[0].weekday(),
            0,  # is_holiday
            forecast['yhat'].mean(),  # previous_sales
            100  # stock_level (example value)
        ]])
        promo_impact = self.promotion_impact_model.predict(promo_features)[0]
        
        # Generate daily predictions
        daily_predictions = []
        for i in range(days):
            prediction = {
                "date": future_dates[i].strftime("%Y-%m-%d"),
                "sales_forecast": self.to_native(forecast['yhat'].iloc[i]),
                "lower_bound": self.to_native(forecast['yhat_lower'].iloc[i]),
                "upper_bound": self.to_native(forecast['yhat_upper'].iloc[i])
            }
            daily_predictions.append(prediction)
        
        # Calculate key metrics and insights
        avg_forecast = float(forecast['yhat'].mean())
        trend = float(forecast['trend'].diff().mean())
        uncertainty = float((forecast['yhat_upper'] - forecast['yhat_lower']).mean() / avg_forecast)
        seasonality = float((forecast['yearly'] + forecast['weekly']).abs().mean() / avg_forecast)
        
        insights = {
            "trend_analysis": {
                "direction": "up" if trend > 0 else "down",
                "strength": self.to_native(abs(trend) / avg_forecast),
                "confidence": self.to_native(1 - uncertainty)
            },
            "price_optimization": {
                "current_price": self.to_native(product_data['price']),
                "optimal_price": self.to_native(optimal_price),
                "potential_uplift": self.to_native((optimal_price - product_data['price']) / product_data['price'] * 100)
            },
            "promotion_impact": {
                "recommended": bool(promo_impact > 0.1),
                "expected_uplift": self.to_native(promo_impact * 100),
                "best_timing": "weekends" if seasonality > 0.1 else "any"
            },
            "market_insights": {
                "seasonality_strength": self.to_native(seasonality),
                "demand_stability": self.to_native(1 - uncertainty),
                "growth_potential": self.to_native(trend / avg_forecast * 100)
            }
        }
        
        return daily_predictions, insights

    def save_models(self, path: str = 'models'):
        """Save all models to disk"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save Prophet model parameters and history
        model_params = {
            'growth': self.prophet_model.growth,
            'n_changepoints': self.prophet_model.n_changepoints,
            'changepoint_range': self.prophet_model.changepoint_range,
            'yearly_seasonality': self.prophet_model.yearly_seasonality,
            'weekly_seasonality': self.prophet_model.weekly_seasonality,
            'daily_seasonality': self.prophet_model.daily_seasonality,
            'seasonality_mode': self.prophet_model.seasonality_mode,
            'seasonality_prior_scale': self.prophet_model.seasonality_prior_scale,
            'changepoint_prior_scale': self.prophet_model.changepoint_prior_scale,
            'holidays_prior_scale': self.prophet_model.holidays_prior_scale,
            'interval_width': self.prophet_model.interval_width,
            'uncertainty_samples': self.prophet_model.uncertainty_samples
        }
        
        with open(f'{path}/prophet_params.json', 'w') as f:
            json.dump(model_params, f)
        
        # Save the trained Prophet model's history
        if hasattr(self.prophet_model, 'history') and self.prophet_model.history is not None:
            self.prophet_model.history.to_csv(f'{path}/prophet_history.csv', index=False)
        
        # Save other models using joblib
        joblib.dump(self.price_impact_model, f'{path}/price_impact_model.joblib')
        joblib.dump(self.promotion_impact_model, f'{path}/promotion_impact_model.joblib')
        joblib.dump(self.scaler, f'{path}/scaler.joblib')

    def load_models(self, path: str = 'models'):
        """Load all models from disk"""
        try:
            # Load Prophet model parameters and recreate the model
            with open(f'{path}/prophet_params.json', 'r') as f:
                model_params = json.load(f)
            
            self.prophet_model = Prophet(
                growth=model_params['growth'],
                n_changepoints=model_params['n_changepoints'],
                changepoint_range=model_params['changepoint_range'],
                yearly_seasonality=model_params['yearly_seasonality'],
                weekly_seasonality=model_params['weekly_seasonality'],
                daily_seasonality=model_params['daily_seasonality'],
                seasonality_mode=model_params['seasonality_mode'],
                seasonality_prior_scale=model_params['seasonality_prior_scale'],
                changepoint_prior_scale=model_params['changepoint_prior_scale'],
                holidays_prior_scale=model_params['holidays_prior_scale'],
                interval_width=model_params['interval_width'],
                uncertainty_samples=model_params['uncertainty_samples']
            )
            
            # Add regressors back
            self.prophet_model.add_regressor('price')
            self.prophet_model.add_regressor('is_promotion')
            
            # Load the history if it exists and refit
            if os.path.exists(f'{path}/prophet_history.csv'):
                history_df = pd.read_csv(f'{path}/prophet_history.csv')
                self.prophet_model.fit(history_df)
            
            # Load other models
            self.price_impact_model = joblib.load(f'{path}/price_impact_model.joblib')
            self.promotion_impact_model = joblib.load(f'{path}/promotion_impact_model.joblib')
            self.scaler = joblib.load(f'{path}/scaler.joblib')
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
