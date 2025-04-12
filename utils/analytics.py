from datetime import datetime, timedelta
import numpy as np
from scipy import stats

def calculate_trend_analysis(data_points, confidence_level=0.95):
    """Calculate trend analysis with statistical confidence"""
    if not data_points:
        return {
            'direction': 'stable',
            'confidence': 0,
            'slope': 0,
            'r_squared': 0
        }
    
    x = np.arange(len(data_points))
    y = np.array(data_points)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    trend_direction = 'increasing' if slope > 0 else 'decreasing'
    confidence = (1 - p_value) * 100  # Convert p-value to confidence percentage
    
    return {
        'direction': trend_direction,
        'confidence': round(confidence, 2),
        'slope': slope,
        'r_squared': r_value ** 2
    }

def calculate_peak_hours(interactions):
    """Determine peak hours based on interaction data"""
    if not interactions:
        return []
    
    hour_counts = {}
    for interaction in interactions:
        hour = interaction['timestamp'].hour
        hour_counts[hour] = hour_counts.get(hour, 0) + 1
    
    # Find the top 3 peak hours
    peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return [f"{hour:02d}:00-{(hour+1):02d}:00" for hour, _ in peak_hours]

def calculate_customer_behavior(interactions, time_window=7):
    """Analyze customer behavior patterns"""
    if not interactions:
        return {
            'conversion_rate': 0,
            'avg_duration': 0,
            'total_interactions': 0
        }
    
    total_interactions = len(interactions)
    high_interest_count = sum(1 for i in interactions if i.get('duration', 0) >= 10)
    conversion_rate = (high_interest_count / total_interactions) * 100 if total_interactions > 0 else 0
    
    durations = [i.get('duration', 0) for i in interactions]
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    return {
        'conversion_rate': round(conversion_rate, 2),
        'avg_duration': round(avg_duration, 2),
        'total_interactions': total_interactions
    }

def generate_time_series_data(interactions, days=7):
    """Generate time series data for visualization"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Initialize data structure
    data = {
        'labels': [],
        'engagement': [],
        'interest': [],
        'crowd': []
    }
    
    # Generate daily data points
    current_date = start_date
    while current_date <= end_date:
        next_date = current_date + timedelta(days=1)
        
        # Filter interactions for current day
        daily_interactions = [
            i for i in interactions 
            if current_date <= i['timestamp'] < next_date
        ]
        
        # Calculate metrics
        if daily_interactions:
            engagement = sum(i.get('confidence', 0) for i in daily_interactions) / len(daily_interactions)
            interest_rate = sum(1 for i in daily_interactions if i.get('duration', 0) >= 5) / len(daily_interactions) * 100
            avg_crowd = sum(i.get('total_people_in_frame', 0) for i in daily_interactions) / len(daily_interactions)
        else:
            engagement = 0
            interest_rate = 0
            avg_crowd = 0
        
        # Add data points
        data['labels'].append(current_date.strftime('%Y-%m-%d'))
        data['engagement'].append(round(engagement, 2))
        data['interest'].append(round(interest_rate, 2))
        data['crowd'].append(round(avg_crowd, 2))
        
        current_date = next_date
    
    return data

def calculate_advanced_metrics(interactions, time_window=7):
    """Calculate advanced metrics for predictions"""
    if not interactions:
        # Return default structure with empty/zero values
        return {
            'time_series': {
                'labels': [],
                'engagement': [],
                'interest': [],
                'crowd': []
            },
            'trends': {
                'engagement': {'direction': 'stable', 'confidence': 0, 'slope': 0, 'r_squared': 0},
                'interest': {'direction': 'stable', 'confidence': 0, 'slope': 0, 'r_squared': 0},
                'crowd': {'direction': 'stable', 'confidence': 0, 'slope': 0, 'r_squared': 0}
            },
            'behavior': {
                'conversion_rate': 0,
                'avg_duration': 0,
                'total_interactions': 0
            },
            'peak_hours': []
        }
    
    # Time series data for trends
    time_series = generate_time_series_data(interactions, time_window)
    
    # Calculate trends
    engagement_trend = calculate_trend_analysis(time_series['engagement'])
    interest_trend = calculate_trend_analysis(time_series['interest'])
    crowd_trend = calculate_trend_analysis(time_series['crowd'])
    
    # Calculate other metrics
    behavior = calculate_customer_behavior(interactions)
    peak_hours = calculate_peak_hours(interactions)
    
    return {
        'time_series': time_series,
        'trends': {
            'engagement': engagement_trend,
            'interest': interest_trend,
            'crowd': crowd_trend
        },
        'behavior': behavior,
        'peak_hours': peak_hours
    }
