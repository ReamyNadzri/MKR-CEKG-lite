# app.py (Final Version with Top-3 Fallback & Optimizations)

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
import PIL.Image

# --- NEW: MongoDB Imports ---
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    
# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")
    MONGO_DB_NAME = "kuih_db"
    
    if not MONGO_DB_PASSWORD:
        logging.warning("MONGO_DB_PASSWORD environment variable not set. Database connection will fail.")
        MONGO_URI = None 
    else:
        MONGO_URI = f"mongodb+srv://kuihdb:{MONGO_DB_PASSWORD}@kuihdb.rcqmsst.mongodb.net/?appName=kuihdb"

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
model_loaded = False

RESEARCH_CLASSES = [
    'Akok', 'Cek Mek Molek','Ketayap', 'Kole Kacang', 'Kuih Bakar', 
    'Kuih Lapis', 'Kuih Lompang', 'Kuih Qasidah', 'Onde-onde', 
    'Pulut Sekaya', 'Seri Muka'
]

# --- MongoDB Globals ---
client = None
db = None
db_connection_ok = False

# --- Gemini AI Configuration ---
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
    if not GEMINI_AVAILABLE:
        return {"error": "Gemini AI is not configured."}

    try:
        logger.info(f"Starting Gemini Vision pre-analysis for: {image_path}")
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_VISION_JSON_SCHEMA
        )
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config) # Updated model name for better availability
        
        class_list_str = ", ".join(research_list)
        prompt = f"""
        You are a Malaysian food expert. Analyze this image and respond only in the required JSON format.
        The list of kuih in my research scope is: [{class_list_str}]

        Your tasks:
        1. Is this a Malaysian kuih?
        2. If NO: Set is_kuih to false, kuih_name to null, is_in_research_scope to null, and provide a brief reason.
        3. If YES: Set is_kuih to true and identify its common 'kuih_name'.
        4. Then, check if this 'kuih_name' is in my research scope list.
        5. If it IS in the scope: Set 'is_in_research_scope' to true and 'estimated_calories' to null.
        6. If it is NOT in the scope: Set 'is_in_research_scope' to false and provide an 'estimated_calories' per piece.
        """
        img = PIL.Image.open(image_path)
        response = model.generate_content([prompt, img])
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Gemini Vision prediction failed: {e}")
        return {"error": f"AI analysis failed: {e}"}

# --- Database Helpers ---
def init_db():
    global client, db, db_connection_ok
    if not MONGO_AVAILABLE: return
    mongo_uri = app.config.get("MONGO_URI")
    if not mongo_uri:
        db_connection_ok = False
        return
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client[Config.MONGO_DB_NAME]
        db_connection_ok = True
        logger.info("MongoDB Atlas connection successful.")
    except ConnectionFailure as e:
        logger.error(f"MongoDB Atlas connection FAILED: {e}")
        db_connection_ok = False

def log_prediction_history(kuih_name, calories):
    if not db_connection_ok: return
    try:
        db.prediction_history.insert_one({
            "kuih_name": kuih_name,
            "calories": str(calories) if calories is not None else 'N/A',
            "timestamp": datetime.now()
        })
    except Exception as e:
        logger.error(f"Failed to log history: {e}")

def get_kuih_details_from_db(kuih_name):
    if not db_connection_ok: return None
    try:
        result = db.calories.find_one({"kuih_name": kuih_name})
        if result:
            return {
                'kuih_name': result.get('kuih_name'),
                'calories': int(result.get('calories')) if result.get('calories') is not None else 'N/A',
                'weight': str(result.get('weight', 'N/A'))
            }
    except Exception as e:
        logger.error(f"DB error fetching details: {e}")
    return None

def save_feedback_to_db(predicted_label, is_correct, actual_label=None, image_filename=None):
    if not db_connection_ok: return False
    image_db_path = None
    if image_filename:
        correct_label = actual_label if not is_correct else predicted_label
        if correct_label:
            image_db_path = os.path.join(Config.FEEDBACK_FOLDER, correct_label, image_filename)
    try:
        db.feedback_log.insert_one({
            "predicted_label": predicted_label,
            "is_correct": 1 if is_correct else 0,
            "actual_label": actual_label,
            "timestamp": datetime.now(),
            "image_path": image_db_path
        })
        return True
    except Exception as err:
        logger.error(f"DB error saving feedback: {err}")
        return False

def get_feedback_stats():
    stats = {'total': 0, 'accuracy': 0}
    if not db_connection_ok: return stats
    try:
        total = db.feedback_log.count_documents({})
        correct = db.feedback_log.count_documents({"is_correct": 1})
        stats['total'] = total
        stats['accuracy'] = (correct / total * 100) if total > 0 else 0
    except Exception as err:
        logger.error(f"DB error stats: {err}")
    return stats

def get_available_classes_from_db():
    if not db_connection_ok: return class_labels
    try:
        classes = db.calories.distinct("kuih_name")
        classes.sort()
        return classes if classes else class_labels
    except Exception as e:
        return class_labels

# --- Model & Prediction Logic ---
def load_trained_model():
    global model, model_loaded
    try:
        if os.path.exists(Config.MODEL_PATH):
            model = load_model(Config.MODEL_PATH)
            model_loaded = True
            logger.info("Model loaded.")
    except Exception as e:
        logger.error(f"Model load failed: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def predict_kuih(image_path):
    """Returns top 3 predictions as list of (label, score) tuples"""
    if not model_loaded: return [("Model Error", 0.0)]
    try:
        img = load_img(image_path, target_size=Config.TARGET_SIZE)
        img_arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
        preds = model.predict(img_arr, verbose=0)
        
        # Get indices of top 3 predictions
        top_3_indices = np.argsort(preds[0])[-3:][::-1]
        top_results = [(class_labels[i], float(preds[0][i])) for i in top_3_indices]
        return top_results
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return [("Prediction Error", 0.0)]

# --- Initialization ---
init_db()
load_trained_model()

# --- Routes ---
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
        'is_gemini_prediction': False
    }

    if not model_loaded or not GEMINI_AVAILABLE:
        return render_template('index.html', error="System Error: Model or AI unavailable.", **render_args), 503

    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template('index.html', error="No file selected.", **render_args), 400
    
    # Note: File validation happens here, but image is already resized by client JS 
    # so it often comes in as 'blob' or the original name.
    
    try:
        fname = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(fpath)

        # 1. Gemini Vision Check
        ai_result = predict_with_gemini_vision(fpath, RESEARCH_CLASSES)

        if "error" in ai_result:
            return render_template('index.html', error=ai_result['error'], **render_args), 500

        # 2. Not a Kuih
        if not ai_result.get('is_kuih'):
            if os.path.exists(fpath): os.remove(fpath)
            return render_template('index.html', error=ai_result.get('reason', 'Not a recognized kuih.'), **render_args), 400

        # 3. Kuih OUTSIDE Scope (Use AI Result)
        elif not ai_result.get('is_in_research_scope'):
            kuih_name = ai_result.get('kuih_name', 'Unknown Kuih')
            calories = ai_result.get('estimated_calories', 'N/A')
            log_prediction_history(kuih_name, calories)

            render_args.update({
                'success': True,
                'kuih_name': kuih_name,
                'confidence': "100% (AI Vision)",
                'confidence_value': 1.0,
                'top_predictions': [(kuih_name, 1.0)],
                'image_path': fname,
                'request_feedback': False,
                'calories': calories,
                'weight': 'N/A',
                'is_gemini_prediction': True
            })
            return render_template('index.html', **render_args)

        # 4. Kuih INSIDE Scope (Use Local CNN)
        else:
            predictions = predict_kuih(fpath)
            kuih_name, conf = predictions[0]

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
                'top_predictions': predictions,
                'image_path': fname,
                'request_feedback': conf < Config.MIN_CONFIDENCE_THRESHOLD,
                'is_gemini_prediction': False
            })

            if details: render_args.update(details)
            else: render_args.update({'calories': 'N/A', 'error_message': f"No details for {kuih_name}"})

            return render_template('index.html', **render_args)

    except Exception as e:
        logger.error(f"Predict route error: {e}")
        return render_template('index.html', error="Server error during prediction.", **render_args), 500

@app.route('/submit_feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    if not data: return jsonify({'success': False, 'message': 'No data'}), 400
    saved = save_feedback_to_db(data.get('predicted_label'), data.get('is_correct'), data.get('actual_label'), data.get('image_path'))
    return jsonify({'success': saved, 'message': 'Feedback saved!' if saved else 'Database error.'})

@app.route('/gemini-info', methods=['POST'])
def get_gemini_info():
    if not GEMINI_AVAILABLE: return jsonify({"error": "AI service is not configured."}), 503
    try:
        data = request.get_json()
        kuih_name = data.get('kuih')
        generation_config = genai.GenerationConfig(response_mime_type="application/json", response_schema=GEMINI_JSON_SCHEMA)
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
        prompt = f"Provide estimated calories, description, and a fun fact for Malaysian kuih: {kuih_name}."
        response = model.generate_content(prompt)
        return jsonify(json.loads(response.text))
    except Exception as e:
        logger.error(f"Gemini Info Error: {e}")
        return jsonify({"error": "Failed to get AI insights."}), 500

@app.route('/api/history')
def get_history():
    if db_connection_ok:
        try:
            return jsonify(list(db.prediction_history.find({}, {"_id": 0}).sort("timestamp", -1).limit(50)))
        except: pass
    return jsonify([])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)