from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import numpy as np
from datetime import datetime, timedelta
import pymongo
import os
from dotenv import load_dotenv
import time
import logging
from pymongo.errors import ConnectionFailure
from urllib.parse import quote_plus
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=True)

# Global variable to control camera thread
camera_active = False
current_camera_index = 0

# MongoDB Atlas connection with retry logic
def connect_to_mongodb(max_retries=3, retry_delay=5):
    username = quote_plus("aswin")
    password = quote_plus("aswin")
    for attempt in range(max_retries):
        try:
            client = pymongo.MongoClient(
                f"mongodb+srv://{username}:{password}@cluster0.4bgll.mongodb.net/sales_forecasting",
                tls=True,
                tlsAllowInvalidCertificates=True,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                maxPoolSize=50,
                retryWrites=True
            )
            # Test connection
            client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas!")
            print("Successfully connected to MongoDB Atlas!")
            return client
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} - Failed to connect to MongoDB Atlas: {e}")
            print(f"Attempt {attempt + 1}/{max_retries} - Failed to connect to MongoDB Atlas: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return None

# Initialize MongoDB connection
client = connect_to_mongodb()
if client:
    try:
        db = client.sales_forecasting
        interactions_collection = db.customer_interactions
        # Test the connection by counting documents
        doc_count = interactions_collection.count_documents({})
        logger.info(f"Successfully connected to MongoDB. Found {doc_count} documents in customer_interactions collection")
        print(f"Successfully connected to MongoDB. Found {doc_count} documents in customer_interactions collection")
    except Exception as e:
        logger.error(f"Error accessing MongoDB collection: {e}")
        print(f"Error accessing MongoDB collection: {e}")
        db = None
        interactions_collection = None
else:
    logger.error("Failed to establish MongoDB connection after all retries")
    print("Failed to establish MongoDB connection after all retries")
    db = None
    interactions_collection = None

def store_interaction(product, position, duration, confidence, total_people):
    """Store high-interest (green) interactions in MongoDB Atlas with total people count"""
    if interactions_collection:
        try:
            interaction = {
                "timestamp": datetime.now(),
                "product": product,
                "position": position,
                "duration": duration,
                "confidence": confidence,
                "interest_level": "high",
                "total_people_in_frame": total_people,
                "crowd_density": "high" if total_people >= 3 else "medium" if total_people == 2 else "low"
            }
            result = interactions_collection.insert_one(interaction)
            logger.info(f"Stored high-interest interaction for {product} with {total_people} people in frame")
            print(f"Stored high-interest interaction for {product} with {total_people} people in frame")
            return True
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
            print(f"Failed to store interaction: {e}")
            return False
    return False

class PersonTracker:
    def __init__(self):
        # Load YOLO model for better person detection
        try:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            weights_path = os.path.join(model_dir, 'yolov3.weights')
            config_path = os.path.join(model_dir, 'yolov3.cfg')
            self.net = cv2.dnn.readNet(weights_path, config_path)
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.positions = {}
            self.last_detection_time = {}
            self.stored_positions = set()  # Track which positions have been stored
            
            # Load class names
            self.classes = []
            with open(os.path.join(model_dir, "coco.names"), "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            logger.info("Successfully initialized YOLO model")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            print(f"Failed to initialize YOLO model: {e}")
            raise

    def detect_and_track(self, frame):
        if frame is None:
            logger.error("Received empty frame")
            return None, []
            
        height, width = frame.shape[:2]
        
        # YOLO detection
        try:
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
            
            # Detection information
            class_ids = []
            confidences = []
            boxes = []
            
            # Process detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Only process person detections with high confidence
                    if confidence > 0.5 and class_id == 0:  # 0 is person in COCO
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-max suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            current_time = time.time()
            detections = []
            mid_point = width // 2
            total_people = len(indexes) if len(indexes) > 0 else 0
            
            if len(indexes) > 0:
                indexes = indexes.flatten()
                for i in indexes:
                    x, y, w, h = boxes[i]
                    center_x = x + w//2
                    position = "left" if center_x < mid_point else "right"
                    product = "Lays" if position == "left" else "Chocolate Cake"
                    
                    # Update tracking
                    if position not in self.positions:
                        self.positions[position] = {
                            "start_time": current_time,
                            "color": "red",
                            "product": product,
                            "confidence": confidences[i]
                        }
                        self.last_detection_time[position] = current_time
                        logger.info(f"New person detected at {position} position")
                    else:
                        self.last_detection_time[position] = current_time
                        duration = current_time - self.positions[position]["start_time"]
                        
                        # Update color and store data
                        if duration >= 10 and self.positions[position]["color"] != "green":
                            self.positions[position]["color"] = "green"
                            # Only store green (high interest) interactions
                            if position not in self.stored_positions:
                                if store_interaction(product, position, duration, confidences[i], total_people):
                                    self.stored_positions.add(position)
                                    logger.info(f"Stored high-interest interaction for {product}")
                        
                        elif duration >= 5 and self.positions[position]["color"] == "red":
                            self.positions[position]["color"] = "yellow"
                    
                    # Draw detection box and label
                    color = {
                        "red": (0, 0, 255),
                        "yellow": (0, 255, 255),
                        "green": (0, 255, 0)
                    }[self.positions[position]["color"]]
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{product} ({self.positions[position]['color']}) - {confidences[i]:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add total people count to the frame
                    people_label = f"Total People: {total_people}"
                    cv2.putText(frame, people_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    detections.append({
                        "position": position,
                        "product": product,
                        "color": self.positions[position]["color"],
                        "confidence": confidences[i],
                        "box": (x, y, w, h),
                        "total_people": total_people
                    })
            
            # Clean up old positions
            for pos in list(self.positions.keys()):
                if current_time - self.last_detection_time[pos] > 2:
                    del self.positions[pos]
                    del self.last_detection_time[pos]
                    if pos in self.stored_positions:
                        self.stored_positions.remove(pos)
            
            return frame, detections
            
        except Exception as e:
            logger.error(f"Error in detect_and_track: {e}")
            return frame, []

# Initialize tracker
tracker = PersonTracker()

def get_available_cameras():
    """Find all available camera devices"""
    available_cameras = []
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to confirm it's working
            ret, _ = cap.read()
            if ret:
                # Get camera name/description if possible
                name = f"Camera {i}"
                available_cameras.append({"index": i, "name": name})
            cap.release()
    return available_cameras

@app.route('/api/cameras', methods=['GET'])
def list_cameras():
    """List all available cameras"""
    try:
        cameras = get_available_cameras()
        return jsonify({
            "success": True,
            "cameras": cameras,
            "message": f"Found {len(cameras)} camera(s)"
        })
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/connect-camera', methods=['POST'])
def connect_camera():
    """Connect to a specific camera"""
    global current_camera_index
    try:
        data = request.get_json()
        camera_index = data.get('camera_index', 0)
        
        # Test camera connection
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_index}")
            return jsonify({"error": f"Failed to open camera {camera_index}"}), 500
            
        current_camera_index = camera_index
        cap.release()
        return jsonify({
            "success": True,
            "message": f"Camera {camera_index} connected successfully"
        }), 200
    except Exception as e:
        logger.error(f"Camera connection error: {e}")
        return jsonify({"error": str(e)}), 500

@socketio.on('stop_detection')
def handle_stop_detection():
    """Stop the camera detection"""
    global camera_active
    logger.info("Stopping detection...")
    print("Stopping detection...")
    camera_active = False
    socketio.emit('detection_stopped', {'message': 'Detection stopped successfully'})

@socketio.on('start_detection')
def handle_detection():
    """Start the camera detection"""
    global camera_active
    logger.info("Starting detection...")
    print("Starting detection...")
    camera_active = True
    
    try:
        cap = cv2.VideoCapture(current_camera_index)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            socketio.emit('error', {'message': 'Failed to open camera'})
            return
            
        # Set camera properties for HD resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
        
        # Get actual camera resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera resolution: {actual_width}x{actual_height}")
        
        # Emit camera info to frontend
        socketio.emit('camera_info', {
            'width': actual_width,
            'height': actual_height,
            'fps': int(cap.get(cv2.CAP_PROP_FPS))
        })
        
        while camera_active:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error("Failed to read from camera")
                socketio.emit('error', {'message': 'Failed to read from camera'})
                break
            
            # Process frame and get detections
            processed_frame, detections = tracker.detect_and_track(frame)
            if processed_frame is None:
                continue
                
            # Emit detections to frontend
            for detection in detections:
                socketio.emit('detection', {
                    'product': detection['product'],
                    'position': detection['position'],
                    'color': detection['color'],
                    'confidence': detection['confidence'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_people': detection['total_people']
                })
            
            # Optimize frame encoding
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Balanced quality
            _, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
            frame_bytes = buffer.tobytes()
            socketio.emit('frame', {'frame': frame_bytes})
            
            # Adaptive delay based on processing time
            time.sleep(0.01)  # Reduced delay for better responsiveness
            
    except Exception as e:
        logger.error(f"Error in detection loop: {e}")
        print(f"Error in detection loop: {e}")
        socketio.emit('error', {'message': f'Detection error: {str(e)}'})
    finally:
        if 'cap' in locals():
            cap.release()
        camera_active = False
        socketio.emit('detection_stopped', {'message': 'Detection stopped'})

@app.route('/api/interactions', methods=['GET'])
def get_interactions():
    if not interactions_collection:
        return jsonify({"error": "MongoDB not connected"}), 503
        
    try:
        product = request.args.get('product')
        query = {"product": product} if product else {}
        interactions = list(interactions_collection.find(query, {"_id": 0}).sort("timestamp", -1))
        return jsonify(interactions)
    except Exception as e:
        logger.error(f"Failed to get interactions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/crowd-analytics', methods=['GET'])
def get_crowd_analytics():
    if not interactions_collection:
        return jsonify({"error": "MongoDB not connected"}), 503
        
    try:
        # Get date range from query parameters or default to last 24 hours
        end_date = datetime.now()
        start_date = request.args.get('start_date', (end_date - timedelta(days=1)).strftime('%Y-%m-%d'))
        
        pipeline = [
            {
                "$match": {
                    "timestamp": {
                        "$gte": datetime.strptime(start_date, '%Y-%m-%d'),
                        "$lte": end_date
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "product": "$product",
                        "hour": {"$hour": "$timestamp"},
                        "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}
                    },
                    "avg_people": {"$avg": "$total_people_in_frame"},
                    "max_people": {"$max": "$total_people_in_frame"},
                    "interest_count": {"$sum": 1},
                    "avg_duration": {"$avg": "$duration"}
                }
            },
            {
                "$sort": {"_id.date": 1, "_id.hour": 1}
            }
        ]
        
        analytics = list(interactions_collection.aggregate(pipeline))
        
        return jsonify({
            "success": True,
            "data": analytics,
            "message": "Successfully retrieved crowd analytics"
        })
        
    except Exception as e:
        logger.error(f"Failed to get crowd analytics: {e}")
        return jsonify({"error": str(e)}), 500

def calculate_product_metrics(product_name, start_date=None, end_date=None):
    """Calculate comprehensive metrics for a product based on historical data"""
    if not start_date:
        start_date = datetime.now() - timedelta(days=7)  # Default to last 7 days
    if not end_date:
        end_date = datetime.now()

    try:
        if not db:
            logger.error("MongoDB not connected")
            return None

        pipeline = [
            {
                "$match": {
                    "product": product_name,
                    "timestamp": {
                        "$gte": start_date,
                        "$lte": end_date
                    }
                }
            },
            {
                "$group": {
                    "_id": "$product",
                    "total_interactions": {"$sum": 1},
                    "avg_duration": {"$avg": "$duration"},
                    "avg_confidence": {"$avg": "$confidence"},
                    "avg_people": {"$avg": "$total_people_in_frame"},
                    "high_interest_count": {
                        "$sum": {"$cond": [{"$eq": ["$interest_level", "high"]}, 1, 0]}
                    },
                    "position_front": {
                        "$sum": {"$cond": [{"$eq": ["$position", "front"]}, 1, 0]}
                    },
                    "peak_crowd": {"$max": "$total_people_in_frame"},
                    "total_duration": {"$sum": "$duration"}
                }
            }
        ]

        logger.info(f"Executing pipeline for product: {product_name}")
        result = list(db.customer_interactions.aggregate(pipeline))
        logger.info(f"Pipeline result: {result}")
        
        metrics = next(iter(result), None)
        
        if metrics:
            total_interactions = metrics['total_interactions']
            metrics['interest_rate'] = (metrics['high_interest_count'] / total_interactions) * 100 if total_interactions > 0 else 0
            metrics['front_position_rate'] = (metrics['position_front'] / total_interactions) * 100 if total_interactions > 0 else 0
            metrics['engagement_score'] = calculate_engagement_score(metrics)
            logger.info(f"Calculated metrics for {product_name}: {metrics}")
            return metrics
        else:
            logger.warning(f"No metrics found for product: {product_name}")
            return {
                "total_interactions": 0,
                "avg_duration": 0,
                "avg_confidence": 0,
                "avg_people": 0,
                "high_interest_count": 0,
                "position_front": 0,
                "peak_crowd": 0,
                "total_duration": 0,
                "interest_rate": 0,
                "front_position_rate": 0,
                "engagement_score": 0
            }
            
    except Exception as e:
        logger.error(f"Error calculating metrics for {product_name}: {e}")
        return None

def calculate_engagement_score(metrics):
    """Calculate an overall engagement score based on multiple factors"""
    weights = {
        'interest_rate': 0.3,
        'front_position_rate': 0.2,
        'avg_duration': 0.2,
        'avg_confidence': 0.15,
        'avg_people': 0.15
    }
    
    # Normalize duration (assuming max duration of 60 seconds)
    normalized_duration = min(metrics['avg_duration'] / 60.0, 1.0)
    
    score = (
        (metrics['interest_rate'] / 100.0 * weights['interest_rate']) +
        (metrics['front_position_rate'] / 100.0 * weights['front_position_rate']) +
        (normalized_duration * weights['avg_duration']) +
        (metrics['avg_confidence'] * weights['avg_confidence']) +
        (min(metrics['avg_people'] / 10.0, 1.0) * weights['avg_people'])
    ) * 100
    
    return round(score, 2)

from utils.analytics import calculate_advanced_metrics

@app.route('/api/product-predictions', methods=['GET'])
def get_product_predictions():
    """Get predictions and comparison for products with advanced analytics"""
    try:
        # Get query parameters
        products = request.args.getlist('products')  # Can be multiple products
        days = int(request.args.get('days', 7))
        
        if not products:
            # Default to all available products if none specified
            products = ['Lays', 'Chocolate Cake']
            
        if not db:
            return jsonify({"error": "Database connection not available"}), 503
            
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        
        results = {}
        for product in products:
            # Get interactions for this product
            interactions = list(db.customer_interactions.find({
                'product': product,
                'timestamp': {'$gte': start_date, '$lte': end_date}
            }))
            
            # Calculate advanced metrics
            advanced_metrics = calculate_advanced_metrics(interactions, days)
            if not advanced_metrics:
                logger.error(f"Failed to calculate advanced metrics for {product}")
                return jsonify({"error": f"Failed to calculate metrics for {product}"}), 500
            
            # Calculate base metrics
            metrics = calculate_product_metrics(product, start_date, end_date)
            if not metrics:
                logger.error(f"Failed to calculate base metrics for {product}")
                return jsonify({"error": f"Failed to calculate metrics for {product}"}), 500
            
            if metrics and advanced_metrics:
                results[product] = {
                    'metrics': {
                        'total_interactions': metrics['total_interactions'],
                        'average_duration': round(metrics['avg_duration'], 2),
                        'engagement_score': round(calculate_engagement_score(metrics), 2),
                        'interest_rate': round(metrics['interest_rate'], 2),
                        'average_crowd': round(metrics['avg_people'], 2),
                        'peak_crowd': metrics['peak_crowd'],
                        'front_position_rate': round(metrics['front_position_rate'], 2)
                    },
                    'predictions': {
                        'trend_direction': advanced_metrics['trends']['engagement']['direction'],
                        'trend_confidence': advanced_metrics['trends']['engagement']['confidence'],
                        'trend_description': f"Engagement is {advanced_metrics['trends']['engagement']['direction']} with {advanced_metrics['trends']['engagement']['confidence']}% confidence",
                        'peak_hours': advanced_metrics['peak_hours'],
                        'customer_behavior_insights': f"Based on {advanced_metrics['behavior']['total_interactions']} interactions, customers show {advanced_metrics['behavior']['conversion_rate']}% conversion rate",
                        'avg_interaction_duration': advanced_metrics['behavior']['avg_duration'],
                        'conversion_rate': advanced_metrics['behavior']['conversion_rate'],
                        'engagement_trend': 'increasing' if metrics['engagement_score'] > 70 else 'stable' if metrics['engagement_score'] > 40 else 'decreasing',
                        'crowd_impact': 'high' if metrics['avg_people'] > 5 else 'medium' if metrics['avg_people'] > 2 else 'low'
                    },
                    'time_series': advanced_metrics['time_series']
                }
        
        # Compare products if multiple are selected
        if len(products) > 1:
            results['comparison'] = compare_products(results)
        
        return jsonify(results)
        
    except Exception as e:
        error_msg = f"Error in get_product_predictions: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)  # This will log the full stack trace
        return jsonify({"error": error_msg}), 500

def compare_products(results):
    """Compare metrics between products"""
    products = list(results.keys())
    if len(products) < 2:
        return None
        
    comparison = {
        "better_engagement": max(products, key=lambda p: results[p]["metrics"]["engagement_score"]),
        "better_interest": max(products, key=lambda p: results[p]["metrics"]["interest_rate"]),
        "higher_crowd": max(products, key=lambda p: results[p]["metrics"]["average_crowd"]),
        "longer_duration": max(products, key=lambda p: results[p]["metrics"]["average_duration"]),
        "summary": []
    }
    
    # Generate detailed comparison insights
    for p1 in products:
        for p2 in products:
            if p1 < p2:  # Compare each pair only once
                eng_diff = results[p1]["metrics"]["engagement_score"] - results[p2]["metrics"]["engagement_score"]
                if abs(eng_diff) > 10:  # Only show significant differences
                    better = p1 if eng_diff > 0 else p2
                    worse = p2 if eng_diff > 0 else p1
                    comparison["summary"].append(
                        f"{better} shows {abs(round(eng_diff, 1))}% better engagement than {worse}"
                    )
    
    return comparison

if __name__ == '__main__':
    logger.info("Starting the application...")
    print("Starting the application...")
    try:
        socketio.run(app, debug=True, port=5000, host='0.0.0.0')
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
        if client:
            client.close()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Application error: {e}")
        if client:
            client.close()
