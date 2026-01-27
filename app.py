# app.py (Consolidated Version with Gemini AI + History + Calculator + MONGODB ATLAS)

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import logging
import uuid
from datetime import datetime
import shutil
import json
import google.generativeai as genai
import PIL.Image  # <-- ADD THIS IMPORT
import google.genai as genai_new # Explicit import for new SDK
# from google.genai import types # We will access types via genai_new.types

# --- NEW: MongoDB Imports ---
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

# --- NEW: Redis Imports for Background Tasks ---
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Background poster generation will be disabled.")

import threading
import time
    
# --- Logging (Moved up to be used by Config) ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    # --- UPDATED: MongoDB Atlas Configuration ---
    MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")
    MONGO_DB_NAME = "kuih_db"
    
    if not MONGO_DB_PASSWORD:
        logging.warning("MONGO_DB_PASSWORD environment variable not set. Database connection will fail.")
        MONGO_URI = None # Set to None to cause a graceful failure
    else:
        # Build the full URI from your template
        MONGO_URI = f"mongodb+srv://kuihdb:{MONGO_DB_PASSWORD}@kuihdb.rcqmsst.mongodb.net/?appName=kuihdb"

    # --- Other Config (Unchanged) ---
    UPLOAD_FOLDER = 'uploads'
    FEEDBACK_FOLDER = 'feedback_images'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    MODEL_PATH = 'kuih_recognition_model.keras'
    TARGET_SIZE = (224, 224)
    MIN_CONFIDENCE_THRESHOLD = 0.7

# --- Flask App Setup ---
app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.FEEDBACK_FOLDER, exist_ok=True)

# --- Global Variables ---
model = None
class_labels = ['Akok', 'Cek Mek Molek','Ketayap', 'Kole Kacang', 'Kuih Bakar', 'Kuih Lapis', 'Kuih Lompang', 'Kuih Qasidah', 'Onde-onde', 'Pulut Sekaya', 'Seri Muka']
metrics = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1-Score': 0.0}
model_loaded = False

# --- ADD THIS (use your exact model class names) ---
RESEARCH_CLASSES = [
    'Akok', 'Cek Mek Molek','Ketayap', 'Kole Kacang', 'Kuih Bakar', 
    'Kuih Lapis', 'Kuih Lompang', 'Kuih Qasidah', 'Onde-onde', 
    'Pulut Sekaya', 'Seri Muka'
]

# --- NEW: MongoDB Globals ---
client = None
db = None
db_connection_ok = False

# --- NEW: Redis Globals ---
redis_client = None
redis_connection_ok = False

# --- NEW: Poster Generation Quota Configuration ---
POSTER_QUOTA_LIMIT = 2  # Number of posters allowed per time window
POSTER_QUOTA_WINDOW = 10800  # 3 hours in seconds (3 * 60 * 60)
POSTER_UNLOCK_CODE = "CSP650"  # Unlock code to bypass quota

# --- NEW: Gemini AI Configuration (Unchanged) ---
try:
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        logger.warning("GEMINI_API_KEY environment variable not set. Gemini features will be disabled.")
        GEMINI_AVAILABLE = False
    else:
        genai.configure(api_key=API_KEY)
        GEMINI_AVAILABLE = True
except Exception as e:
    logger.error(f"Error configuring Gemini AI: {e}")
    GEMINI_AVAILABLE = False

GEMINI_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
    "estimatedcalories":{
            "type": "STRING",
            "description": "A short brief, estimated the calories based on the standard size (g), piece and sources."
        },
        "othersname": {
            "type": "STRING",
            "description": "Other names or local variations of the kuih."
        },
        "description": {
            "type": "STRING",
            "description": "A brief, engaging 2-3 sentence description of the kuih."
        },
        "fun_fact": {
            "type": "STRING",
            "description": "A single interesting fun fact about the kuih's history, ingredients, or cultural significance"
        }
    },
    "required": ["estimatedcalories","othersname","description", "fun_fact"]
}

GEMINI_VISION_JSON_SCHEMA = {
  "type": "OBJECT",
  "properties": {
    "is_kuih": {
      "type": "BOOLEAN",
      "description": "True if the image is a Malaysian kuih, False otherwise."
    },
    "kuih_name": {
      "type": "STRING",
      "description": "The common name of the kuih (e.g., 'Kuih Lapis', 'Karipap'). Null if is_kuih is False."
    },
    "is_in_research_scope": {
      "type": "BOOLEAN",
      "description": "True if the kuih_name is found in the provided list, False otherwise. Null if is_kuih is False."
    },
    "estimated_calories": {
      "type": "STRING",
      "description": "Estimated calories per piece (e.g., '90-110 kcal'). ONLY provide this if is_in_research_scope is False. Otherwise, set to Null."
    },
    "reason": {
      "type": "STRING",
      "description": "If is_kuih is False, briefly state what the image is (e.g., 'This is a car.')"
    }
  },
  "required": ["is_kuih", "kuih_name", "is_in_research_scope", "estimated_calories", "reason"]
}

def predict_with_gemini_vision(image_path, research_list):
    """
    Analyzes an image using Gemini Vision to check if it's a kuih 
    and if it's in the research scope.
    """
    if not GEMINI_AVAILABLE:
        return {"error": "Gemini AI is not configured."}

    try:
        logger.info(f"Starting Gemini Vision pre-analysis for: {image_path}")

        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_VISION_JSON_SCHEMA
        )
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config=generation_config
        )

        # Convert the list to a string for the prompt
        class_list_str = ", ".join(research_list)

        prompt = f"""
        You are a Malaysian food expert. Analyze this image and respond only in the required JSON format.
        The list of kuih in my research scope is: [{class_list_str}]

        Your tasks:
        1.  Is this a Malaysian kuih?
        2.  If NO: Set is_kuih to false, kuih_name to null, is_in_research_scope to null, and provide a brief reason.
        3.  If YES: Set is_kuih to true and identify its common 'kuih_name'.
        4.  Then, check if this 'kuih_name' is in my research scope list.
        5.  If it IS in the scope: Set 'is_in_research_scope' to true and 'estimated_calories' to null.
        6.  If it is NOT in the scope: Set 'is_in_research_scope' to false and provide an 'estimated_calories' per piece.
        """

        img = PIL.Image.open(image_path)

        response = model.generate_content([prompt, img])
        response_data = json.loads(response.text)

        logger.info(f"Gemini Vision pre-analysis response: {response_data}")
        return response_data

    except Exception as e:
        logger.error(f"Gemini Vision prediction failed: {e}")
        return {"error": f"AI analysis failed: {e}"}

# --- UPDATED: Database Helpers (MongoDB Atlas) ---
def init_db():
    """Initializes the MongoDB client and database object."""
    global client, db, db_connection_ok
    if not MONGO_AVAILABLE:
        logger.warning("Pymongo not found. MongoDB features disabled.")
        return

    # Get the fully constructed URI from the app config
    mongo_uri = app.config.get("MONGO_URI")

    if not mongo_uri:
        logger.error("MONGO_URI not configured. Did you set the MONGO_DB_PASSWORD environment variable?")
        db_connection_ok = False
        return

    try:
        # Create a single client. serverSelectionTimeoutMS checks connection within 5 secs.
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info() # Force connection test
        db = client[Config.MONGO_DB_NAME]
        db_connection_ok = True
        logger.info("MongoDB Atlas connection successful.")
    except ConnectionFailure as e:
        logger.error(f"MongoDB Atlas connection FAILED: {e}")
        db_connection_ok = False
        client = None
        db = None

# --- History Logging Function (MongoDB) ---
def log_prediction_history(kuih_name, calories):
    """Logs a successful prediction to the history collection."""
    if not db_connection_ok: return
    try:
        history_collection = db.prediction_history
        cal_val = str(calories) if calories is not None else 'N/A'
        history_collection.insert_one({
            "kuih_name": kuih_name,
            "calories": cal_val,
            "timestamp": datetime.now()
        })
    except Exception as e:
        logger.error(f"Failed to log history to MongoDB: {e}")

# --- Get Kuih Details (MongoDB) ---
def get_kuih_details_from_db(kuih_name):
    if not db_connection_ok: return None
    details = None
    try:
        calories_collection = db.calories
        result = calories_collection.find_one({"kuih_name": kuih_name})
        
        if result:
            details = {
                'kuih_name': result.get('kuih_name'),
                'calories': int(result.get('calories')) if result.get('calories') is not None else 'N/A',
                'size_servings': result.get('size_servings', 'N/A'),
                'weight': str(result.get('weight', 'N/A')),
                'references': result.get('references', 'N/A'),
                'links': result.get('links', 'N/A')
            }
    except Exception as e:
        logger.error(f"DB error fetching details from MongoDB: {e}")
    return details

# --- Save Feedback (MongoDB) ---
def save_feedback_to_db(predicted_label, is_correct, actual_label=None, image_filename=None):
    if not db_connection_ok: return False
    
    image_db_path = None
    if image_filename:
        correct_label_for_path = actual_label if not is_correct else predicted_label
        if correct_label_for_path:
            image_db_path = os.path.join(Config.FEEDBACK_FOLDER, correct_label_for_path, image_filename)

    try:
        feedback_collection = db.feedback_log
        feedback_collection.insert_one({
            "predicted_label": predicted_label,
            "is_correct": 1 if is_correct else 0,
            "actual_label": actual_label,
            "timestamp": datetime.now(),
            "image_path": image_db_path
        })
        logger.info(f"Feedback saved to DB: Image='{image_filename}', Predicted='{predicted_label}', Correct={is_correct}, Actual='{actual_label}'")
        return True
    except Exception as err:
        logger.error(f"DB error saving feedback to MongoDB: {err}")
        return False

# --- Get Feedback Stats (MongoDB) ---
def get_feedback_stats():
    stats = {'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0}
    if not db_connection_ok: return stats
    
    try:
        feedback_collection = db.feedback_log
        stats['total'] = feedback_collection.count_documents({})
        if stats['total'] > 0:
            stats['correct'] = feedback_collection.count_documents({"is_correct": 1})
            stats['incorrect'] = stats['total'] - stats['correct']
            stats['accuracy'] = (stats['correct'] / stats['total'] * 100)
    except Exception as err:
        logger.error(f"DB error getting feedback stats from MongoDB: {err}")
    return stats

# --- Get Available Classes (MongoDB) ---
def get_available_classes_from_db():
    if not db_connection_ok: return class_labels
    try:
        calories_collection = db.calories
        classes = calories_collection.distinct("kuih_name")
        classes.sort()
        return classes if classes else class_labels
    except Exception as e:
        logger.error(f"DB error getting classes from MongoDB: {e}")
        return class_labels

# --- Model & Utils (TensorFlow 2.20 Compatible) ---
def load_trained_model():
    global model, model_loaded
    try:
        if os.path.exists(Config.MODEL_PATH):
            logger.info(f"Loading model from {Config.MODEL_PATH}")
            # Try loading with compile=False for better TF 2.20 compatibility
            try:
                model = load_model(Config.MODEL_PATH, compile=False)
                # Recompile with compatible optimizer
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("Model loaded successfully with recompilation")
            except Exception as compile_error:
                # Fallback to standard loading
                logger.warning(f"Recompile approach failed: {compile_error}. Trying standard load...")
                model = load_model(Config.MODEL_PATH)
                logger.info("Model loaded with standard method")
            
            model_loaded = True
            logger.info(f"Model loaded successfully. TensorFlow version: {tf.__version__}")
        else:
            logger.error(f"Model file not found at {Config.MODEL_PATH}")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        logger.error(f"TensorFlow version: {tf.__version__}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def predict_kuih(image_path):
    if not model_loaded: return "Model Error", 0.0
    try:
        img = load_img(image_path, target_size=Config.TARGET_SIZE)
        img_arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
        preds = model.predict(img_arr, verbose=0)
        idx = np.argmax(preds[0])
        return class_labels[idx], float(preds[0][idx])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return "Prediction Error", 0.0

# --- NEW: Redis Initialization ---
def init_redis():
    """Initializes the Redis client for job queue."""
    global redis_client, redis_connection_ok
    if not REDIS_AVAILABLE:
        logger.warning("Redis not available. Poster generation will run synchronously (may timeout on Railway).")
        return
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        redis_connection_ok = True
        logger.info(f"Redis connection successful: {redis_url}")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        redis_connection_ok = False
        redis_client = None

# --- NEW: Poster Quota Management Functions ---
def get_user_ip():
    """Get the user's IP address from the request."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0]
    return request.remote_addr

def get_poster_quota_status(ip_address):
    """
    Check poster generation quota for an IP address.
    Returns: dict with 'allowed', 'remaining', 'unlocked', 'reset_time'
    """
    if not redis_connection_ok:
        # Fallback: allow without quota if Redis unavailable
        return {'allowed': True, 'remaining': POSTER_QUOTA_LIMIT, 'unlocked': False, 'reset_time': None}
    
    quota_key = f"poster_quota:{ip_address}"
    quota_data = redis_client.hgetall(quota_key)
    
    if not quota_data:
        # No quota record - user hasn't generated yet
        return {'allowed': True, 'remaining': POSTER_QUOTA_LIMIT, 'unlocked': False, 'reset_time': None}
    
    # Check if unlocked
    if quota_data.get('unlocked') == 'true':
        return {'allowed': True, 'remaining': 999, 'unlocked': True, 'reset_time': None}
    
    # Check remaining count
    count = int(quota_data.get('count', 0))
    remaining = POSTER_QUOTA_LIMIT - count
    
    if remaining > 0:
        expires_at = quota_data.get('expires_at')
        return {'allowed': True, 'remaining': remaining, 'unlocked': False, 'reset_time': expires_at}
    else:
        # Quota exhausted
        expires_at = quota_data.get('expires_at')
        return {'allowed': False, 'remaining': 0, 'unlocked': False, 'reset_time': expires_at}

def use_poster_quota(ip_address):
    """
    Increment the poster generation count for an IP address.
    Returns: True if quota used successfully, False if quota exceeded
    """
    if not redis_connection_ok:
        return True  # Allow if Redis unavailable
    
    quota_key = f"poster_quota:{ip_address}"
    quota_data = redis_client.hgetall(quota_key)
    
    # Check if unlocked
    if quota_data and quota_data.get('unlocked') == 'true':
        return True  # Unlimited access
    
    if not quota_data:
        # First use - create quota record
        now = datetime.now()
        expires_at = now + timedelta(seconds=POSTER_QUOTA_WINDOW)
        redis_client.hmset(quota_key, {
            'count': 1,
            'unlocked': 'false',
            'first_use': now.isoformat(),
            'expires_at': expires_at.isoformat()
        })
        redis_client.expire(quota_key, POSTER_QUOTA_WINDOW)
        return True
    
    # Increment count
    count = int(quota_data.get('count', 0))
    if count >= POSTER_QUOTA_LIMIT:
        return False  # Quota exceeded
    
    redis_client.hincrby(quota_key, 'count', 1)
    return True

def unlock_poster_quota(ip_address, unlock_code):
    """
    Unlock unlimited poster generation for an IP address.
    Returns: True if unlock successful, False if code invalid
    """
    if unlock_code != POSTER_UNLOCK_CODE:
        return False
    
    if not redis_connection_ok:
        logger.warning("Cannot unlock quota - Redis not available")
        return False
    
    quota_key = f"poster_quota:{ip_address}"
    # Set unlocked flag (expires after 24 hours)
    redis_client.hmset(quota_key, {
        'count': 0,
        'unlocked': 'true',
        'unlocked_at': datetime.now().isoformat()
    })
    redis_client.expire(quota_key, 86400)  # 24 hours
    logger.info(f"Poster quota unlocked for IP: {ip_address}")
    return True


# --- NEW: Background Worker for Poster Generation ---
def process_poster_jobs():
    """Background worker that processes poster generation jobs."""
    if not redis_connection_ok:
        return
    
    logger.info("Poster job worker started")
    while True:
        try:
            # Get all job keys
            job_keys = redis_client.keys("poster_job:*")
            
            for job_key in job_keys:
                job_data = redis_client.hgetall(job_key)
                
                if job_data.get('status') == 'PENDING':
                    job_id = job_key.split(':')[1]
                    logger.info(f"Processing job {job_id}")
                    
                    # Update status to PROCESSING
                    redis_client.hset(job_key, 'status', 'PROCESSING')
                    
                    try:
                        # Perform the actual poster generation
                        kuih_name = job_data.get('kuih_name')
                        image_filename = job_data.get('image_filename')
                        calories = job_data.get('calories', 'N/A')
                        
                        # Resolve image path
                        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_filename))
                        
                        if not os.path.exists(image_path):
                            raise Exception("Image file not found")
                        
                        # Call Gemini API
                        client_gemini = genai_new.Client(api_key=API_KEY)
                        
                        prompt = f"""
                        Ultra-clean modern recipe infographic. Showcase {kuih_name} in a visually appealing finished form—sliced, plated, or
portioned—floating slightly in perspective or angled view. Arrange ingredients,
steps, and tips around the dish in a dynamic editorial layout, not restricted
to top-down. Ingredients Section: Include icons or mini illustrations for each
ingredient with quantities. Arrange them in clusters, lists, or circular flows
connected visually to the dish. Steps Section: Show preparation steps with
numbered panels, arrows, or lines, forming a logical flow around the main dish.
Include small cooking icons (knife, pan, oven, timer) where helpful. Additional
Info (optional): Total calories, prep/cook time, servings, spice
level—displayed as clean bubbles or badges near the dish. Visual Style:
Editorial infographic meets lifestyle food photography. Vibrant, natural food
colors, subtle drop shadows, clean vector icons, modern typography, soft
gradients or glassmorphism for step panels. Accent colors can highlight key
info (calories: {calories}, prep time). Composition Guidelines: Finished meal as hero
visual (perspective or angled) Ingredients and steps flow dynamically around
the dish Clear visual hierarchy: dish > steps > ingredients > optional
stats Enough negative space to keep design airy and readable Lighting &
Background: Soft, natural studio lighting, minimal textured or gradient
background for premium editorial feel. Output: 1080×1080, ultra-crisp,
social-feed optimized, no watermark.`
                        """
                        
                        response = client_gemini.models.generate_content(
                            model="gemini-3-pro-image-preview",
                            contents=[
                                prompt,
                                PIL.Image.open(image_path),
                            ],
                            config=genai_new.types.GenerateContentConfig(
                                response_modalities=['IMAGE'],
                                image_config=genai_new.types.ImageConfig(
                                    aspect_ratio="1:1",
                                    image_size="2K"
                                ),
                            )
                        )
                        
                        # Extract image from response
                        image_base64 = None
                        for part in response.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                import base64
                                raw_data = part.inline_data.data
                                image_base64 = base64.b64encode(raw_data).decode('utf-8')
                                mime_type = part.inline_data.mime_type or "image/png"
                                image_base64 = f"data:{mime_type};base64,{image_base64}"
                                break
                            
                            if hasattr(part, 'as_image'):
                                try:
                                    img = part.as_image()
                                    if img:
                                        import io
                                        import base64
                                        buffered = io.BytesIO()
                                        img.save(buffered, format="PNG")
                                        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                        image_base64 = f"data:image/png;base64,{img_str}"
                                        break
                                except:
                                    pass
                        
                        if image_base64:
                            # Success - update job with result
                            redis_client.hset(job_key, 'status', 'COMPLETED')
                            redis_client.hset(job_key, 'result', json.dumps({"image_base64": image_base64}))
                            logger.info(f"Job {job_id} completed successfully")
                        else:
                            raise Exception("No image generated by AI")
                        
                    except Exception as e:
                        logger.error(f"Job {job_id} failed: {e}")
                        redis_client.hset(job_key, 'status', 'FAILED')
                        redis_client.hset(job_key, 'error', str(e))
            
            # Sleep between polls
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(5)

# --- Initialization ---
init_db() # <-- Initialize DB connection on startup
init_redis() # <-- Initialize Redis connection
load_trained_model()

# Start background worker if Redis is available
if redis_connection_ok:
    worker_thread = threading.Thread(target=process_poster_jobs, daemon=True)
    worker_thread.start()
    logger.info("Background poster worker thread started")

# --- Routes (Unchanged) ---
@app.route('/')
def home():
    return render_template('index.html',
                         model_loaded=model_loaded,
                         db_connection_ok=db_connection_ok,
                         feedback_stats=get_feedback_stats(),
                         available_classes=get_available_classes_from_db())

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], secure_filename(filename))

@app.route('/predict', methods=['POST'])
def handle_predict():
    render_args = {
        'model_loaded': model_loaded,
        'db_connection_ok': db_connection_ok,
        'feedback_stats': get_feedback_stats(),
        'available_classes': get_available_classes_from_db(),
        'is_gemini_prediction': False  # Default to false (local model)
    }

    if not model_loaded or not GEMINI_AVAILABLE:
        return render_template('index.html', error="System error: A required component (Model or AI) is not configured.", **render_args), 503

    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template('index.html', error="No file selected.", **render_args), 400
    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type.", **render_args), 400

    try:
        # --- Step 1: Save the file ---
        fname = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(fpath)

        # --- Step 2: Always pre-analyze with Gemini Vision ---
        ai_result = predict_with_gemini_vision(fpath, RESEARCH_CLASSES)

        if "error" in ai_result:
            return render_template('index.html', error=ai_result['error'], **render_args), 500

        # --- Step 3: Check AI results and decide workflow ---

        # Condition 1: Not a kuih
        if not ai_result.get('is_kuih'):
            logger.info("AI determined image is NOT a kuih.")
            if os.path.exists(fpath): os.remove(fpath) # Clean up
            return render_template('index.html', error=ai_result.get('reason', 'This does not appear to be a kuih.'), **render_args), 400

        # Condition 2: Is a kuih, but NOT in research scope (Use AI results)
        elif not ai_result.get('is_in_research_scope'):
            logger.info("AI determined kuih is OUTSIDE research scope. Using AI prediction.")
            kuih_name = ai_result.get('kuih_name', 'Unknown Kuih')
            calories = ai_result.get('estimated_calories', 'N/A')

            log_prediction_history(kuih_name, calories)

            render_args.update({
                'success': True,
                'kuih_name': kuih_name,
                'confidence': "100% (AI)",
                'confidence_value': 1.0,
                'image_path': fname,
                'request_feedback': False,
                'calories': calories,
                'weight': 'N/A',
                'is_gemini_prediction': True # This will hide the feedback buttons
            })

            return render_template('index.html', **render_args)

        # Condition 3: Is a kuih AND is in research scope (Use local CNN model)
        else:
            logger.info("AI determined kuih is INSIDE research scope. Using local CNN.")
            kuih_name, conf = predict_kuih(fpath)

            if "Error" in kuih_name:
                 if os.path.exists(fpath): os.remove(fpath)
                 return render_template('index.html', error=kuih_name, **render_args), 500

            details = get_kuih_details_from_db(kuih_name)
            calories_to_log = details['calories'] if details else 'N/A'
            log_prediction_history(kuih_name, calories_to_log)

            render_args.update({
                'success': True,
                'kuih_name': kuih_name,
                'confidence': f"{conf*100:.2f}%",
                'confidence_value': conf,
                'image_path': fname,
                'request_feedback': conf < Config.MIN_CONFIDENCE_THRESHOLD,
                'is_gemini_prediction': False # This will show the feedback buttons
            })

            if details:
                render_args.update(details)
            else:
                 render_args.update({'calories': 'N/A', 'error_message': f"No details for {kuih_name}"})

            return render_template('index.html', **render_args)

    except Exception as e:
        logger.error(f"Predict route error: {e}")
        return render_template('index.html', error="Server error during prediction.", **render_args), 500

@app.route('/submit_feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    if not data: return jsonify({'success': False, 'message': 'No data'}), 400
    
    saved = save_feedback_to_db(
        data.get('predicted_label'),
        data.get('is_correct'),
        data.get('actual_label'),
        data.get('image_path')
    )
    return jsonify({'success': saved, 'message': 'Feedback saved!' if saved else 'Database error.'})

# --- NEW: Gemini AI Route ---
@app.route('/gemini-info', methods=['POST'])
def get_gemini_info():
    """Get description and fun fact from Gemini."""
    if not GEMINI_AVAILABLE:
        logger.error("Gemini route called, but Gemini is not available (check API key).")
        return jsonify({"error": "AI service is not configured."}), 503

    try:
        data = request.get_json()
        kuih_name = data.get('kuih')
        if not kuih_name:
            return jsonify({"error": "No kuih name provided."}), 400
        
        logger.info(f"Requesting Gemini info for: {kuih_name}")

        # --- Set up the model with the JSON schema ---
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_JSON_SCHEMA
        )
        model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-09-2025",
            generation_config=generation_config
        )
        
        # --- Create the prompt ---
        prompt = f"""
        You are a Malaysian food expert. 
        Provide estimation calories, serving and the sources and others names and a 2-3 sentence description and 
        one interesting fun fact for the following Malaysian kuih: {kuih_name}.
        Ensure the fun fact is different from the description.
        """
        
        # --- Call the API ---
        response = model.generate_content(prompt)
        
        # --- Parse the JSON response ---
        # The response.text will be a JSON string that matches the schema
        response_data = json.loads(response.text)
        
        logger.info(f"Successfully got Gemini info for: {kuih_name}")
        return jsonify(response_data)

    except Exception as e:
        logger.exception(f"Error calling Gemini API: {e}")
        return jsonify({"error": "Failed to get AI insights."}), 500
# --- END NEW ---

@app.route('/generate_poster', methods=['POST'])
def generate_poster():
    """Creates a background job for poster generation and returns job_id immediately.
    Falls back to synchronous mode if Redis is unavailable."""
    if not API_KEY:
        return jsonify({"error": "AI service is not configured."}), 503
    
    # --- QUOTA CHECK ---
    user_ip = get_user_ip()
    quota_status = get_poster_quota_status(user_ip)
    
    if not quota_status['allowed']:
        reset_time = quota_status.get('reset_time')
        if reset_time:
            try:
                reset_dt = datetime.fromisoformat(reset_time)
                time_left = reset_dt - datetime.now()
                minutes_left = int(time_left.total_seconds() / 60)
                hours = minutes_left // 60
                mins = minutes_left % 60
                time_msg = f"{hours}h {mins}m" if hours > 0 else f"{mins} minutes"
            except:
                time_msg = "a while"
        else:
            time_msg = "3 hours"
        
        return jsonify({
            "error": f"Poster generation limit reached ({POSTER_QUOTA_LIMIT} per 3 hours). Try again in {time_msg} or unlock with code CSP650.",
            "quota_exceeded": True,
            "remaining": 0,
            "reset_time": reset_time
        }), 429
    
    # If Redis is not available, fallback to synchronous mode
    if not redis_connection_ok:
        logger.warning("Redis not available, falling back to synchronous poster generation")
        # Redirect to synchronous endpoint
        return generate_poster_sync()

    try:
        data = request.get_json()
        kuih_name = data.get('kuih')
        image_filename = data.get('image_filename')
        calories = data.get('calories', 'N/A')

        if not kuih_name:
            return jsonify({"error": "No kuih name provided."}), 400
        
        # Verify image exists
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_filename)) if image_filename else None
        if not image_path or not os.path.exists(image_path):
            return jsonify({"error": "Original image not found for poster generation."}), 400

        # Create job in Redis
        job_id = uuid.uuid4().hex
        job_key = f"poster_job:{job_id}"
        
        job_data = {
            'job_id': job_id,
            'status': 'PENDING',
            'kuih_name': kuih_name,
            'image_filename': image_filename,
            'calories': calories,
            'created_at': datetime.now().isoformat()
        }
        
        redis_client.hmset(job_key, job_data)
        redis_client.expire(job_key, 3600)  # Expire after 1 hour
        
        # Use quota for this generation
        use_poster_quota(user_ip)
        
        logger.info(f"Created poster generation job {job_id} for {kuih_name}")
        return jsonify({"success": True, "job_id": job_id, "remaining": quota_status['remaining'] - 1})
        
    except Exception as e:
        logger.exception(f"Error creating poster job: {e}")
        return jsonify({"error": f"Failed to create job: {str(e)}"}), 500

@app.route('/poster_status/<job_id>', methods=['GET'])
def get_poster_status(job_id):
    """Checks the status of a poster generation job."""
    if not redis_connection_ok:
        return jsonify({"error": "Job system not available"}), 503
    
    try:
        job_key = f"poster_job:{job_id}"
        job_data = redis_client.hgetall(job_key)
        
        if not job_data:
            return jsonify({"error": "Job not found"}), 404
        
        status = job_data.get('status')
        
        response = {
            'job_id': job_id,
            'status': status
        }
        
        if status == 'COMPLETED':
            result_json = job_data.get('result')
            logger.info(f"Job {job_id} completed. Result field exists: {result_json is not None}")
            if result_json:
                try:
                    result = json.loads(result_json)
                    logger.info(f"Result parsed. Has image_base64: {'image_base64' in result}")
                    if 'image_base64' in result:
                        img_size = len(result['image_base64'])
                        logger.info(f"Image base64 size: {img_size} bytes")
                    response['result'] = result
                except json.JSONDecodeError as je:
                    logger.error(f"Failed to parse result JSON: {je}")
                    response['result'] = {}
            else:
                logger.warning(f"Job {job_id} marked COMPLETED but no result field in Redis")
        elif status == 'FAILED':
            response['error'] = job_data.get('error', 'Unknown error')
        
        return jsonify(response)
    except Exception as e:
        logger.exception(f"Error checking job status: {e}")
        return jsonify({"error": str(e)}), 500

# --- NEW: Poster Quota Endpoints ---
@app.route('/poster_quota', methods=['GET'])
def get_quota():
    """Returns the current poster generation quota status for the user."""
    user_ip = get_user_ip()
    quota_status = get_poster_quota_status(user_ip)
    return jsonify(quota_status)

@app.route('/unlock_poster', methods=['POST'])
def unlock_poster():
    """Unlocks unlimited poster generation with the correct code."""
    data = request.get_json()
    unlock_code = data.get('code', '').strip()
    
    if not unlock_code:
        return jsonify({"success": False, "error": "No unlock code provided"}), 400
    
    user_ip = get_user_ip()
    success = unlock_poster_quota(user_ip, unlock_code)
    
    if success:
        return jsonify({
            "success": True,
            "message": "Unlock successful! You now have unlimited poster generations for 24 hours."
        })
    else:
        return jsonify({
            "success": False,
            "error": "Invalid unlock code. Please try again."
        }), 403


# --- SYNCHRONOUS POSTER GENERATION (FALLBACK) ---
def generate_poster_sync():
    """Generates a recipe poster using Gemini AI (New SDK) with Image Input - SYNCHRONOUS VERSION.
    Can be called directly via /generate_poster_sync or as fallback when Redis unavailable."""
    # Note: We re-check API Key availability here because the global check was for the old SDK
    # But usually they use the same key.
    if not API_KEY: # Re-using the global API_KEY variable
        return jsonify({"error": "AI service is not configured."}), 503

    try:
        data = request.get_json()
        kuih_name = data.get('kuih')
        
        image_filename = data.get('image_filename') # Get the uploaded filename
        calories = data.get('calories', 'N/A') # Get calories

        if not kuih_name:
            return jsonify({"error": "No kuih name provided."}), 400
        
        # Resolve image path
        image_path = None
        if image_filename:
             # Sanitize just in case, though it comes from our internal logic usually
             image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_filename))
        
        if not image_path or not os.path.exists(image_path):
             return jsonify({"error": "Original image not found for poster generation."}), 400

        logger.info(f"Generating poster SYNCHRONOUSLY for: {kuih_name} using image: {image_filename}")

        # --- Initialize Client (New SDK) ---
        client = genai_new.Client(api_key=API_KEY)

        # --- Construct the Prompt ---
        prompt = f"""
        Ultra-clean modern recipe infographic. Showcase {kuih_name} in a visually appealing finished
        form—sliced, plated, or
portioned—floating slightly in perspective or angled view. Arrange ingredients,
steps, and tips around the dish in a dynamic editorial layout, not restricted
to top-down. Ingredients Section: Include icons or mini illustrations for each
ingredient with quantities. Arrange them in clusters, lists, or circular flows
connected visually to the dish. Steps Section: Show preparation steps with
numbered panels, arrows, or lines, forming a logical flow around the main dish.
Include small cooking icons (knife, pan, oven, timer) where helpful. Additional
Info (optional): Total calories, prep/cook time, servings, spice
level—displayed as clean bubbles or badges near the dish. Visual Style:
Editorial infographic meets lifestyle food photography. Vibrant, natural food
colors, subtle drop shadows, clean vector icons, modern typography, soft
gradients or glassmorphism for step panels. Accent colors can highlight key
info (calories: {calories}, prep time). Composition Guidelines: Finished meal as hero
visual (perspective or angled) Ingredients and steps flow dynamically around
the dish Clear visual hierarchy: dish > steps > ingredients > optional
stats Enough negative space to keep design airy and readable Lighting &
Background: Soft, natural studio lighting, minimal textured or gradient
background for premium editorial feel. Output: 1080×1080, ultra-crisp,
social-feed optimized, no watermark.`
        """

        # --- Verify Image Before Sending ---
        try:
            test_img = PIL.Image.open(image_path)
            logger.info(f"Image loaded successfully: {test_img.format}, {test_img.size}, {test_img.mode}")
        except Exception as e:
            logger.error(f"Failed to open image at {image_path}: {e}")
            return jsonify({"error": f"Could not load image: {str(e)}"}), 500

        # --- Generate Content ---
        logger.info(f"Sending image to Gemini API for poster generation...")
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[
                prompt,
                PIL.Image.open(image_path),  # Open fresh for API
            ],
            config=genai_new.types.GenerateContentConfig(
                response_modalities=['IMAGE'], # Requesting Image
                image_config=genai_new.types.ImageConfig(
                    aspect_ratio="1:1", # Square as requested (1080x1080 implied)
                    image_size="2K"
                ),
            )
        )
        logger.info(f"Received response from Gemini API")
        
        # --- Handle Response ---
        # "for part in response.parts: if image:= part.as_image(): image.save..."
        # We want to return base64.
        
        for part in response.parts:
            # Check for inline_data first (raw bytes) - safer for Base64 conversion
            if hasattr(part, 'inline_data') and part.inline_data:
                 logger.info("Found inline_data in response part.")
                 import base64
                 # data is bytes
                 raw_data = part.inline_data.data
                 logger.info(f"Raw data type: {type(raw_data)}, Size: {len(raw_data) if raw_data else 0} bytes")
                 
                 image_data = base64.b64encode(raw_data).decode('utf-8')
                 mime_type = part.inline_data.mime_type or "image/png"
                 logger.info(f"Returning Base64 image. Mime: {mime_type}")
                 return jsonify({"success": True, "image_base64": f"data:{mime_type};base64,{image_data}"})
                 
            # Fallback: Try as_image() but handle potential custom object issues
            if hasattr(part, 'as_image'):
                 try:
                     logger.info("Trying part.as_image()...")
                     img = part.as_image()
                     if img:
                         import io
                         import base64
                         buffered = io.BytesIO()
                         # Try standard PIL save
                         img.save(buffered, format="PNG")
                         img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                         logger.info("Returning Base64 image via PIL save.")
                         return jsonify({"success": True, "image_base64": f"data:image/png;base64,{img_str}"})
                 except Exception as e:
                     logger.warning(f"Failed to process image via as_image(): {e}")
                     # Continue to next part or text

        # If we got here, maybe only text?
        text_content = ""
        if response.text:
            text_content = response.text
        
        return jsonify({"success": False, "message": "Model returned no image. Text: " + text_content})

    except Exception as e:
        logger.exception(f"Error generating poster: {e}")
        return jsonify({"error": f"Failed to generate poster: {str(e)}"}), 500

from bson import ObjectId # Added for ObjectId handling

# ... (existing imports)

# --- NEW ROUTES FOR TAB NAVIGATION ---
@app.route('/overview')
def overview():
    return render_template('overview.html')

@app.route('/system')
def system():
    return render_template('system.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/history')
def get_history():
    history_data = []
    if db_connection_ok:
        try:
            history_collection = db.prediction_history
            
            # Filter by today (00:00:00 to 23:59:59)
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            query = {"timestamp": {"$gte": today_start}}
            
            # Get records
            cursor = history_collection.find(query).sort("timestamp", -1)
            
            for doc in cursor:
                # Convert ObjectId to string for JSON serialization
                doc['_id'] = str(doc['_id'])
                history_data.append(doc)
                
        except Exception as e:
             logger.error(f"History fetch error from MongoDB: {e}")
    
    return jsonify(history_data)

@app.route('/api/history/<id>', methods=['DELETE'])
def delete_history_item(id):
    if not db_connection_ok:
        return jsonify({'success': False, 'message': 'Database not connected'}), 500
    try:
        history_collection = db.prediction_history
        result = history_collection.delete_one({'_id': ObjectId(id)})
        if result.deleted_count > 0:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Item not found'}), 404
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)