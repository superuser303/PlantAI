import streamlit as st
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import openai
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import base64
from pathlib import Path
import time

openai.api_key = "YOUR_OPENAI_API_KEY"
# Load pre-trained DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

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


# Use of medicine dictionary
use_of_medicine = {
    "Aloevera": "{improve skin and prevent wrinkles,wound healing}",
    "Amla": "{controlling diabetes,hair amazing,losing weight,skin healthy}",
    "Amruta_Balli": "{Immunomodulatory, fever.}",
    "Arali": "{ parts for traditional healing.}",
    "Ashoka": "{Uterine health, menstrual issue.}",
    "Ashwagandha": "{Adaptogen, stress relief.}",
    "Avocado": "{ Nutrient-rich, heart health.}",
    "Bamboo": "{Shoots, traditional cuisine.}",
    "Basale": "{Shoots, traditional cuisine.}",
    "Betel": "{Digestive, chewed with areca nut.}",
    "Betel_Nut": "{Chewing, traditional practices, caution.}",
    "Brahmi": "{Cognitive enhancer, adaptogen}",
    "Castor": "{ Oil for medicinal, cosmetic use}",
    "Curry_Leaf": "{ Flavoring, potential traditional uses.}",
    "Doddapatre": "{ Poultice, skin conditions.}",
    "Ekka": "{Traditional uses, caution for toxicity.}",
    "Ganike": "{Respiratory health, traditional medicine.}",
    "Guava": "{ Vitamin C, digestive benefits}",
    "Geranium": "{ Oil for aromatherapy, skincare.}",
    "Henna": "{ Hair coloring, natural dye.}",
    "Hibiscus": "{Tea for antioxidants, skin health.}",
    "Honge": "{Anti-inflammatory, traditional use.}",
    "Insulin": "{Potential blood sugar regulation, traditional use.}",
    "Jasmine": "{Tea, relaxation, stress relief.}",
    "Lemon": "{Digestive aid, rich in vitamin C.}",
    "Lemon_grass": "{Tea, digestive, calming effects.}",
    "Mango": "{Fruit, traditional uses for health.}",
    "Mint": "{Tea, aids digestion, refreshing flavor.}",
    "Nagadali": "{Traditional uses, potential medicinal purposes.}",
    "Neem": "{ Antibacterial, antifungal, supports skin health.}",
    "Nithyapushpa": "{Calming effects, traditional use.}",
    "Nooni": "{ Oil from seeds, various traditional uses.}",
    "Pappaya": "{ Fruit, leaves, traditional uses.}",
    "Pepper": "{ Spice, potential digestive benefits.}",
    "Pomegranate": "{Antioxidant-rich, heart health.}",
    "Raktachandini": "{Traditional uses, caution for potential toxicity.}",
    "Rose": "{Tea, calming, aromatic effects.}",
    "Sapota": "{Sweet taste, nutritional content.}",
    "Tulasi": "{Tea, immune support, respiratory health.}",
    "Wood_sorel": "{Leaves in salads, some varieties may have medicinal uses.}"
}

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
        </style>
    """, unsafe_allow_html=True)
    
# Load model with caching
@st.cache_resource
def load_prediction_model():
    try:
        model = load_model("//home//santy//DesignProject//Medicinal_Plant.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def add_chatbot():
    # Header for the chatbot
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #064e3b; font-weight: 700;">💬 AI Chatbot</h2>
            <p style="color: #374151; font-size: 1.1rem;">Choose an AI model and start chatting!</p>
        </div>
    """, unsafe_allow_html=True)

    # Model selection option
    model_option = st.selectbox("Choose AI Model:", ["Free AI Model", "ChatGPT"])

    # Initialize chat history in session state if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize chat history ids for DialoGPT if not exists
    if "chat_history_ids" not in st.session_state:
        st.session_state.chat_history_ids = None

    # User input
    user_input = st.text_input(
        "Type your question here...",
        placeholder="Ask me anything...",
        help="Enter your query and click Send",
        label_visibility="collapsed",
    )

    if st.button("Send", use_container_width=True):
        if user_input:
            if model_option == "Free AI Model":
                try:
                    # Use DialoGPT
                    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

                    # Concatenate with chat history if it exists
                    if st.session_state.chat_history_ids is not None:
                        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1)
                    else:
                        bot_input_ids = new_user_input_ids

                    # Generate response
                    chat_history_ids = model.generate(
                        bot_input_ids,
                        max_length=1000,
                        pad_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=3,
                        do_sample=True,
                        top_k=100,
                        top_p=0.7,
                        temperature=0.8
                    )

                    # Update chat history ids
                    st.session_state.chat_history_ids = chat_history_ids

                    # Decode response
                    response = tokenizer.decode(
                        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                        skip_special_tokens=True
                    )

                except Exception as e:
                    response = f"An error occurred: {str(e)}"
                    st.error("Error in processing request")

            elif model_option == "ChatGPT":
                try:
                    # Check if API key is set
                    if not openai.api_key or openai.api_key == "YOUR_OPENAI_API_KEY":
                        st.error("Please set your OpenAI API key in the code")
                        return

                    # Use OpenAI's ChatGPT
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": user_input}]
                    )
                    response = completion.choices[0].message['content']

                except Exception as e:
                    response = f"An error occurred: {str(e)}"
                    st.error("Error in processing request")

            # Append to chat history
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))

    # Display chat history with styling
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(
                f"""<div style='text-align: right; margin: 10px 0; padding: 10px; 
                border-radius: 10px; background-color: #e5e7eb;'>
                <strong>{speaker}:</strong> {message}</div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style='text-align: left; margin: 10px 0; padding: 10px; 
                border-radius: 10px; background-color: #dcfce7;'>
                <strong>{speaker}:</strong> {message}</div>""",
                unsafe_allow_html=True
            )

def predict_class(img):
    try:
        st.session_state['loading'] = True
        img = Image.open(img).convert('RGB')
        img = img.resize((256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        if st.session_state['model'] is None:
            st.session_state['model'] = load_prediction_model()

        if st.session_state['model'] is not None:
            time.sleep(1)
            predictions = st.session_state['model'].predict(img_array)
            predicted_class_index = np.argmax(predictions)
            confidence = float(predictions[0][predicted_class_index]) * 100
            st.session_state['loading'] = False
            return class_labels[predicted_class_index], confidence
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
                uses = uses.strip("{}").split(",")
                for use in uses:
                    st.markdown(f"• {use.strip()}")
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="glass-card" style="text-align: center;">
                    <h3 style="color: #374151;">No Image Uploaded</h3>
                    <p style="color: #6b7280;">Upload an image to see the analysis</p>
                </div>
            """, unsafe_allow_html=True)
    add_chatbot()
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
