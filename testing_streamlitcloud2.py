import gdown
import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import base64
from pathlib import Path
import time

# Initialize session state
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False
if 'loading' not in st.session_state:
    st.session_state['loading'] = False
    
# Set page configuration
st.set_page_config(
    page_title="MediPlant AI | Plant Identification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class labels and other data (moved outside functions)
class_labels = ["Aloevera", "Amla", "Amruta_Balli", "Arali", "Ashoka", "Ashwagandha", "Avocado", "Bamboo", "Basale",
                "Betel", "Betel_Nut", "Brahmi", "Castor", "Curry_Leaf", "Doddapatre", "Ekka", "Ganike", "Guava",
                "Geranium", "Henna", "Hibiscus", "Honge", "Insulin", "Jasmine", "Lemon", "Lemon_grass", "Mango",
                "Mint", "Nagadali", "Neem", "Nithyapushpa", "Nooni", "Pappaya", "Pepper", "Pomegranate",
                "Raktachandini", "Rose", "Sapota", "Tulasi", "Wood_sorel"]

# Methods of preparation dictionary
methods_of_preparation = {
    "Aloevera": "Slit the leaf of an aloe plant lengthwise and remove the gel from the inside, or use a commercial preparation.",
    "Amla": "Eating raw amla and candies or taking amla powder with lukewarm water",
    "Amruta_Balli": "Make a decoction or powder from the stems of Giloy. It is known for its immunomodulatory properties.",
    "Arali": "Various parts like the root bark, leaves, and fruit are used for medicinal purposes. It can be consumed in different forms, including as a decoction.",
    "Ashoka": "Different parts like the bark are used. It's often prepared as a decoction for menstrual and uterine health.",
    "Ashwagandha": "The root is commonly used, and it can be consumed as a powder, capsule, or as a decoction. It is an adaptogen known for its stress-relieving properties.",
    "Avocado": "The fruit is consumed for its nutritional benefits, including healthy fats and vitamins.",
    "Bamboo": "Bamboo shoots are consumed, and some varieties are used in traditional medicine.",
    "Basale": "The leaves are consumed as a leafy vegetable. It's rich in vitamins and minerals.",
    "Betel": "Chewing betel leaves with areca nut is a common practice in some cultures. It's believed to have digestive and stimulant properties.",
    "Betel_Nut": "The nut is often chewed with betel leaves. However, excessive consumption is associated with health risks.",
    "Brahmi": "The leaves are used for enhancing cognitive function. It can be consumed as a powder, in capsules, or as a fresh juice.",
    "Castor": "Castor oil is extracted from the seeds and used for various medicinal and cosmetic purposes.",
    "Curry_Leaf": "Curry leaves are used in cooking for flavor, and they are also consumed for their potential health benefits.",
    "Doddapatre": "The leaves are used in traditional medicine, often as a poultice for skin conditions.",
    "Ekka": "Various parts may be used in traditional medicine. It's important to note that some species of Ekka may have toxic components, and proper identification is crucial.",
    "Ganike": "The leaves are used in traditional medicine, often as a remedy for respiratory issues.",
    "Guava": "Guava fruit is consumed for its high vitamin C content and other health benefits.",
    "Geranium": "Geranium oil is extracted from the leaves and stems and is used in aromatherapy and skincare.",
    "Henna": "Henna leaves are dried and powdered to make a paste used for hair coloring and as a natural dye.",
    "Hibiscus": "Hibiscus flowers are commonly used to make teas, infusions, or extracts. They are rich in antioxidants and can be beneficial for skin and hair health.",
    "Honge": "Various parts of the tree are used traditionally, including the bark and seeds. It's often used for its anti-inflammatory properties.",
    "Insulin": "The leaves are used for their potential blood sugar-lowering properties. They can be consumed fresh or as a tea.",
    "Jasmine": "Jasmine flowers are often used to make aromatic teas or essential oils, known for their calming effects.",
    "Lemon": "Lemon juice is a common remedy for digestive issues, and the fruit is rich in vitamin C. The peel can be used for its essential oil.",
    "Lemon_grass": "Lemon grass is used to make teas and infusions, known for its soothing and digestive properties.",
    "Mango": "Mango fruit is consumed fresh and is rich in vitamins and minerals. Some parts, like the leaves, are also used in traditional medicine.",
    "Mint": "Mint leaves are commonly used to make teas, infusions, or added to dishes for flavor. It's known for its digestive properties.",
    "Nagadali": "Different parts of the plant are used traditionally. It's often prepared as a decoction.",
    "Neem": "Various parts of the neem tree are used, including leaves, bark, and oil. It's known for its antibacterial and antifungal properties.",
    "Nithyapushpa": "The flowers are used in traditional medicine, often for their calming effects.",
    "Nooni": "Different parts of the tree are used traditionally. The oil extracted from the seeds is used for various purposes.",
    "Pappaya": "Consume fruit; leaves traditionally used for certain health benefits.",
    "Pepper": "Spice for flavor; potential digestive and antimicrobial properties.",
    "Pomegranate": "Eat seeds or drink juice for antioxidant benefits.",
    "Raktachandini": "Traditional uses; some parts may be toxic, use caution.",
    "Rose": "Make tea or use petals for calming and aromatic effects.",
    "Sapota": "Consume fruit for its sweet taste and nutritional content.",
    "Tulasi": "Make tea or use leaves for immune support.",
    "Wood_sorel": "Make tea or use leaves for immune support. Use leaves in salads; some varieties contain oxalic acid."
}

    # Original dictionary
use_of_medicine = {
    "Lemon_grass": [
        "Calms the nervous system and reduces anxiety.",
        "Aids digestion and relieves bloating.",
        "Has anti-inflammatory and pain-relieving properties."
    ],
    "Mango": [
        "Rich in vitamins A and C, boosts immune health.",
        "Aids in digestion and improves skin condition."
    ],
    "Mint": [
        "Soothes the digestive system and relieves nausea.",
        "Has a refreshing effect and clears nasal congestion.",
        "Acts as a natural breath freshener."
    ],
    "Nagadali": [
        "Anti-inflammatory properties, used in pain relief.",
        "Supports traditional medicinal practices."
    ],
    "Neem": [
        "Antibacterial and antifungal, supports skin health.",
        "Boosts immunity and helps purify the blood."
    ],
    "Nithyapushpa": [
        "Calming effect, used in stress and anxiety relief.",
        "Promotes mental well-being in traditional medicine."
    ],
    "Nooni": [
        "Anti-inflammatory properties, used for pain relief.",
        "Boosts immune function and overall well-being."
    ],
    "Pappaya": [
        "Aids in digestion with enzymes like papain.",
        "Rich in vitamins and supports skin health.",
        "Used traditionally for wound healing and immunity."
    ],
    "Pepper": [
        "Improves digestion and relieves bloating.",
        "Has antimicrobial properties and boosts metabolism."
    ],
    "Pomegranate": [
        "Rich in antioxidants, supports heart and skin health.",
        "Anti-inflammatory and aids in digestion."
    ],
    "Raktachandini": [
        "Anti-inflammatory and used in pain relief.",
        "Traditional use with caution due to toxicity."
    ],
    "Rose": [
        "Calming effect, used in aromatherapy.",
        "Hydrates skin and promotes relaxation."
    ],
    "Sapota": [
        "Rich in dietary fiber, aids in digestion.",
        "Provides energy and supports skin health."
    ],
    "Tulasi": [
        "Boosts immunity and supports respiratory health.",
        "Anti-inflammatory and used in treating colds."
    ],
    "Wood_sorel": [
        "Rich in vitamin C, used in treating scurvy.",
        "Anti-inflammatory and aids digestion."
    ],
    "Jasmine": [
        "Calms the mind and reduces anxiety.",
        "Used in teas for relaxation and stress relief."
    ],
    "Lemon": [
        "Rich in vitamin C, boosts immune function.",
        "Aids in digestion and supports detoxification.",
        "Promotes skin health and fights free radicals."
    ]
}

# Applying strip to each string in the dictionary
for key in use_of_medicine:
    use_of_medicine[key] = [use.strip() for use in use_of_medicine[key]]

def load_css():
    st.markdown("""
    <style>
    /* Enhanced Header Styles */
    .header-container {
        padding: 2rem 0;
        text-align: center;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        animation: fadeInDown 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }

    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, transparent, #059669, transparent);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        100% { left: 100%; }
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .main-title {
        color: #064e3b;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(120deg, #064e3b, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titlePulse 2s infinite;
    }

    @keyframes titlePulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    } 
        /* Modern Team Members Section */
        .team-section {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 1rem;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(229, 231, 235, 0.5);
        }
        
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .team-member-card {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            padding: 1rem;
            border-radius: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            transition: all 0.3s ease;
            border: 1px solid rgba(5, 150, 105, 0.1);
        }
        
        .team-member-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        
        .member-avatar {
            width: 40px;
            height: 40px;
            background: #059669;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .member-info {
            flex: 1;
        }
        
        .member-name {
            color: #065f46;
            font-weight: 600;
            margin: 0;
            font-size: 1rem;
        }
        
        .member-role {
            color: #059669;
            font-size: 0.85rem;
            margin: 0;
        }
        
        /* Modern Drop Zone */
        .modern-drop-zone {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 2px dashed #059669;
            border-radius: 1rem;
            padding: 3rem 2rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .modern-drop-zone:hover {
            border-color: #065f46;
            transform: scale(1.01);
        }
        
        .modern-drop-zone.dragging {
            background: rgba(240, 253, 244, 0.9);
            border-color: #065f46;
            transform: scale(1.02);
        }
        
        .upload-icon {
            width: 64px;
            height: 64px;
            margin: 0 auto 1rem;
            color: #059669;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .drop-zone-text {
            color: #065f46;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .drop-zone-subtext {
            color: #059669;
            font-size: 0.9rem;
        }
        
        .supported-formats {
            margin-top: 1rem;
            display: flex;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .format-badge {
            background: rgba(5, 150, 105, 0.1);
            color: #059669;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .chat-container {
        background: white;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        padding: 1rem;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        cursor: pointer;
    }
    
    .chat-header:hover {
        background: linear-gradient(135deg, #065f46 0%, #05875f 100%);
    }
    
    .chat-body {
        padding: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .chat-input {
        background: #f3f4f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem;
    }
    
    /* Enhanced Results Section */
    .result-card {
        background: white;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .result-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .confidence-meter {
        background: #e5e7eb;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #059669 0%, #064e3b 100%);
        transition: width 0.5s ease;
    }
    
    .property-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .property-card {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        transition: transform 0.2s ease;
    }
    
    .property-card:hover {
        transform: translateY(-2px);
    }
    
    .expandable-section {
        margin-top: 1rem;
    }
    
    .expandable-header {
        background: #f3f4f6;
        padding: 0.75rem;
        border-radius: 0.5rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .expandable-content {
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 0 0 0.5rem 0.5rem;
        margin-top: 0.25rem;
    }
    /* Animated Confidence Bar Styles */
    .confidence-container {
        margin: 1rem 0;
        padding: 1rem;
        background: white;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .confidence-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        color: #374151;
        font-weight: 500;
    }
    
    .confidence-bar {
        height: 1rem;
        background: #e5e7eb;
        border-radius: 1rem;
        overflow: hidden;
        position: relative;
    }
    
    .confidence-fill {
        height: 100%;
        width: 0;
        border-radius: 1rem;
        background: linear-gradient(90deg, #059669, #064e3b);
        transition: width 1s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .confidence-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            45deg,
            rgba(255,255,255,0.2) 25%,
            transparent 25%,
            transparent 50%,
            rgba(255,255,255,0.2) 50%,
            rgba(255,255,255,0.2) 75%,
            transparent 75%,
            transparent
        );
        background-size: 1rem 1rem;
        animation: shimmer 1s linear infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-1rem); }
        100% { transform: translateX(1rem); }
    }
    
    /* Color variations based on confidence level */
    .confidence-fill.high {
        background: linear-gradient(90deg, #059669, #064e3b);
    }
    
    .confidence-fill.medium {
        background: linear-gradient(90deg, #0ea5e9, #0369a1);
    }
    
    .confidence-fill.low {
        background: linear-gradient(90deg, #f59e0b, #d97706);
    }
     # Add this to your existing CSS in load_css() function
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid rgba(5, 150, 105, 0.3);
        border-top: 4px solid #059669;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .loading-text {
        text-align: center;
        color: #065f46;
        margin-top: 15px;
        font-weight: 600;
    }

    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: rgba(5, 150, 105, 0.05);
        border-radius: 10px;
}           
    </style>

    """, unsafe_allow_html=True)    
# Load model with caching
@st.cache_resource
def load_prediction_model():
    try:
        download_url = "https://drive.google.com/uc?id=17xebXPPkKbQYJjAE0qyxikUjoUY6BNoz"
        model_path = "Medicinal_Plant.h5"
        gdown.download(download_url, model_path, quiet=False)
        model = tf.keras.models.load_model(model_path)
        return model
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded object is not a valid Keras model")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
        
def predict_class(img):
    try:
        # Set loading state to True at the start of prediction
        st.session_state['loading'] = True
        
        img = Image.open(img).convert('RGB')
        img = img.resize((256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        if st.session_state['model'] is None:
            st.session_state['model'] = load_prediction_model()

        if st.session_state['model'] is not None:
            # Optional: Add a slight delay to show loading spinner
            time.sleep(1.5)
            
            predictions = st.session_state['model'].predict(img_array)
            predicted_class_index = np.argmax(predictions)
            confidence = float(predictions[0][predicted_class_index]) * 100
            
            # Set loading state to False after prediction
            st.session_state['loading'] = False
            return class_labels[predicted_class_index], confidence
        
        # If model loading fails
        st.session_state['loading'] = False
        return None, None
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.session_state['loading'] = False
        return None, None
        
def main():
    # Load CSS
        load_css()
    
    # Header with enhanced animation, slogan, and modernized team members
        st.markdown("""
        <div class="header-container">
            <h1 class="main-title">🌿 MediPlant AI</h1>
            <p class="subtitle" style="color: #065f46; font-size: 1.2rem; margin-bottom: 1rem;">
                Advanced medicinal plant identification powered by artificial intelligence. 
                Upload a photo to identify plants and discover their therapeutic properties.
            </p>
            <p class="slogan" style="color: #059669; font-style: italic; font-size: 1.1rem; margin-bottom: 1rem;">
                "Unlocking Nature's Medicine Cabinet with AI"
            </p>
            <div class="guide-info" style="background: rgba(5, 150, 105, 0.1); padding: 0.5rem; border-radius: 0.5rem; display: inline-block;">
                <p style="color: #065f46; margin: 0;">
                    <strong>Project Guide:</strong> Mrs. A Anitharani
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
        st.markdown("""
    <div class="team-section">
        <h4 style="color: #065f46; margin: 0 0 0.5rem 0;">Project Team</h4>
        <div class="team-grid">
            <div class="team-member-card">
                <div class="member-avatar">S</div>
                <div class="member-info">
                    <p class="member-name">Santhoshkumar J</p>
                    <p class="member-role">Team Member</p>
                </div>
            </div>
            <div class="team-member-card">
                <div class="member-avatar">R</div>
                <div class="member-info">
                    <p class="member-name">Raghul M</p>
                    <p class="member-role">Team Member</p>
                </div>
            </div>
            <div class="team-member-card">
                <div class="member-avatar">S</div>
                <div class="member-info">
                    <p class="member-name">Shivam Sinha</p>
                    <p class="member-role">Team Member</p>
                </div>
            </div>
            <div class="team-member-card">
                <div class="member-avatar">K</div>
                <div class="member-info">
                    <p class="member-name">S Keerthika</p>
                    <p class="member-role">Team Member</p>
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])
    
with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload a plant image",
            type=["jpg", "png", "jpeg"],
            help="Supported formats: JPG, PNG, JPEG",
            key="plant_image_uploader"
        )
        
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)
        else:
            st.markdown("""
                <div class="modern-drop-zone" id="custom-drop-zone">
                    <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 15V3m0 12l-4-4m4 4l4-4M2 17l.621 2.485A2 2 0 004.561 21h14.878a2 2 0 001.94-1.515L22 17" stroke-linecap="round" stroke-linejoin="round"/></path>
                    </svg>
                    <h3 class="drop-zone-text">Drop your plant image here</h3>
                    <p class="drop-zone-subtext">or click to browse your files</p>
                    <div class="supported-formats">
                        <span class="format-badge">JPG</span>
                        <span class="format-badge">PNG</span>
                        <span class="format-badge">JPEG</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Add JavaScript for drag and drop functionality
            st.markdown("""
                <script>
                    const dropZone = document.getElementById('custom-drop-zone');
                    
                    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                        dropZone.addEventListener(eventName, preventDefaults, false);
                    });
                    
                    function preventDefaults (e) {
                        e.preventDefault();
                        e.stopPropagation();
                    }
                    
                    ['dragenter', 'dragover'].forEach(eventName => {
                        dropZone.addEventListener(eventName, highlight, false);
                    });
                    
                    ['dragleave', 'drop'].forEach(eventName => {
                        dropZone.addEventListener(eventName, unhighlight, false);
                    });
                    
                    function highlight(e) {
                        dropZone.classList.add('dragging');
                    }
                    
                    function unhighlight(e) {
                        dropZone.classList.remove('dragging');
                    }
                    
                    dropZone.addEventListener('drop', handleDrop, false);
                    
                    function handleDrop(e) {
                        const dt = e.dataTransfer;
                        const files = dt.files;
                        const fileInput = document.querySelector('input[type="file"]');
                        
                        if (fileInput) {
                            fileInput.files = files;
                            fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                    }
                </script>
            """, unsafe_allow_html=True)
with col2:
        if uploaded_file:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Loading Spinner
        if st.session_state['loading']:
            st.markdown("""
                <div class="loading-container">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">Analyzing plant image...</div>
                </div>
            """, unsafe_allow_html=True)    
            
            predicted_class, confidence = predict_class(uploaded_file)
            
            if predicted_class and not st.session_state['loading']:
                st.markdown(f"""
                    <div class="prediction-container">
                        <h2 style="color: #064e3b; margin-bottom: 0.5rem;">
                            {predicted_class}
                        </h2>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence}%"></div>
                        </div>
                        <p style="color: #374151;">Confidence: {confidence:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Method of Preparation
                st.markdown("""
                    <div class="info-section">
                        <h3 class="info-title">📝 Method of Preparation</h3>
                """, unsafe_allow_html=True)
                st.write(methods_of_preparation.get(predicted_class, "No information available"))
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Medicinal Uses
                st.markdown("""
                    <div class="info-section">
                        <h3 class="info-title">💊 Medicinal Uses</h3>
                """, unsafe_allow_html=True)
                uses = use_of_medicine.get(predicted_class, "No information available")
                if isinstance(uses, list):
                    for use in uses:
                        st.markdown(f"• {use}")
                else:
                    st.markdown(f"• {uses}")
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="glass-card" style="text-align: center;">
                    <h3 style="color: #374151;">No Image Uploaded</h3>
                    <p style="color: #6b7280;">Upload an image to see the analysis</p>
                </div>
            """, unsafe_allow_html=True)

    # About section with enhanced animation
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #064e3b; margin-bottom: 1rem;">About MediPlant AI</h3>
            <p style="color: #374151; line-height: 1.6;">
                MediPlant AI uses advanced machine learning to identify medicinal plants and provide detailed information 
                about their traditional uses and preparation methods. Our system can identify 40 different species of 
                medicinal plants commonly used in traditional medicine with high accuracy.
            </p>
            <br>
            <h4 style="color: #064e3b; margin-bottom: 0.5rem;">Features:</h4>
            <ul style="color: #374151; line-height: 1.6;">
                <li>Real-time plant identification</li>
                <li>Detailed preparation methods</li>
                <li>Traditional medicinal uses</li>
                <li>Confidence scoring</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

