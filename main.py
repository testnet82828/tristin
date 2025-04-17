import os
import json
import base64
import numpy as np
import tensorflow as tf
import streamlit as st
import io
from PIL import Image
from supabase import create_client, Client
from datetime import datetime
import google.generativeai as genai
import gdown

# Clear cache to avoid stale model files
st.cache_resource.clear()

# ------------------------- SUPABASE CONFIGURATION -------------------------
SUPABASE_URL = "https://nnvjlzsmqzqmzyobqwfw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5udmpsenNtcXpxbXp5b2Jxd2Z3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzk5Nzc3MDMsImV4cCI6MjA1NTU1MzcwM30.deqXBiufjAoFHeh69_FBkNhx6BDLBoyTcpg-4a72vM4"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------- GOOGLE GENERATIVE AI CONFIGURATION -------------------------
GOOGLE_API_KEY = "AIzaSyCGn_JarFPfoPcNxmB7hDCy1cLgJm8TSYs"
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the generative model with a faster model
chatbot_model = genai.GenerativeModel('gemini-1.5-flash-002')

# ------------------------- BACKGROUND IMAGE FUNCTION -------------------------
def get_image_base64(image_path):
    try:
        image_path = os.path.join(os.path.dirname(__file__), 'images', image_path)
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        st.error(f"Error: Background image '{image_path}' not found. Please check the file path.")
        return ""

# ------------------------- AUTHENTICATION -------------------------
def login_page():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url('data:image/jpg;base64,{get_image_base64('login-image.jfif')}');
                background-size: cover;
                color: white;
            }}
            h1 {{
                color: white;
            }}
            .stTextInput > label, .stSelectbox > label {{
                color: white;
            }}
            div[data-baseweb="input"] > div > input {{
                color: black !important;
                background-color: white;
            }}
            .stButton > button {{
                background-color: black;
                color: white;
                border: 1px solid white;
            }}
            .stButton > button:hover {{
                background-color: #333333;
                color: white;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🌿 Welcome to Plant Disease Classifier")
    choice = st.selectbox("Login / Signup / Continue as Guest", ["Login", "Signup", "Continue as Guest"])

    if choice == "Continue as Guest":
        st.session_state["user"] = "guest"
        st.session_state["user_id"] = None
        st.session_state.pop("access_token", None)
        st.session_state.pop("refresh_token", None)
        st.success("You are using the site as a guest.")
        st.experimental_rerun()
        return

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if choice == "Signup":
        if st.button("Signup"):
            try:
                supabase.auth.sign_up({"email": email, "password": password})
                session = supabase.auth.sign_in_with_password({"email": email, "password": password})
                uuid = session.user.id

                # Combine into a single dictionary
                supabase.table("profiles").insert({
                    "email": email,
                    "uuid": uuid,
                    "password": password  # ⚠️ Not recommended to store raw passwords!
                }).execute()

                st.success("Account created! Please log in.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        if st.button("Login"):
            try:
                session = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state["user"] = session
                st.session_state["user_id"] = session.user.id
                st.session_state["access_token"] = session.session.access_token
                st.session_state["refresh_token"] = session.session.refresh_token
                supabase.auth.set_session(session.session.access_token, session.session.refresh_token)
                st.success(f"Logged in as {session.user.email}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# ------------------------- MODEL CONFIGURATION -------------------------
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")

@st.cache_resource
def download_model():
    model_path = "trained_model/plant_disease_prediction_model.h5"
    if not os.path.exists(model_path):
        os.makedirs("trained_model", exist_ok=True)
        url = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # Replace with your Google Drive file ID
        st.write(f"Downloading model from Google Drive to: {model_path}")
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model from Google Drive: {e}")
            raise
    st.write(f"Model path: {model_path}")
    st.write(f"File exists: {os.path.exists(model_path)}")
    st.write(f"File size: {os.path.getsize(model_path)} bytes")
    return model_path

@st.cache_resource
def load_model(model_path):
    st.write(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    file_size = os.path.getsize(model_path)
    st.write(f"File size: {file_size} bytes")
    if file_size < 10000:  # Check for Git LFS pointer file
        st.error("File is likely a Git LFS pointer, not the actual model.")
        raise ValueError("Invalid model file: Likely a Git LFS pointer.")
    
    try:
        with h5py.File(model_path, 'r') as f:
            st.write("Valid HDF5 file")
    except Exception as e:
        st.error(f"Invalid HDF5 file: {e}")
        raise ValueError(f"Invalid HDF5 file: {e}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        st.write("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

@st.cache_resource
def load_class_indices(class_file_path):
    st.write(f"Attempting to load class indices from: {class_file_path}")
    if not os.path.exists(class_file_path):
        st.error(f"Class indices file not found at: {class_file_path}")
        raise FileNotFoundError(f"Class indices file not found at: {class_file_path}")
    with open(class_file_path, 'r') as f:
        return json.load(f)

working_dir = os.path.dirname(os.path.abspath(__file__))
st.write(f"Current working directory: {working_dir}")
try:
    model_path = download_model()
    st.write("Root directory contents:", os.listdir(working_dir))
    if os.path.exists(f"{working_dir}/trained_model"):
        st.write("Trained model directory contents:", os.listdir(f"{working_dir}/trained_model"))
    else:
        st.write("Trained model directory does not exist, created during download.")
    class_file_path = f"{working_dir}/class_indices.json"
    disease_model = load_model(model_path)
    class_indices = load_class_indices(class_file_path)
except Exception as e:
    st.error(f"Error loading model or class indices: {e}")
    st.stop()

# ------------------------- IMAGE PROCESSING -------------------------
@st.cache_data
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image).resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices.get(str(predicted_class_index), "Unknown")

# ------------------------- HISTORY FUNCTIONS -------------------------
def save_to_history(user_id, image_data, prediction):
    if user_id:
        file_name = f"{user_id}/{datetime.now().isoformat()}.png"
        insert_data = {
            "user_id": user_id,
            "image_path": file_name,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        try:
            auth_user = supabase.auth.get_user()
            if not auth_user:
                st.error("No authenticated user. Session may have expired.")
                return
            st.write(f"Debug: Authenticated user ID: {auth_user.user.id}, Insert user_id: {user_id}")
            if auth_user.user.id != user_id:
                st.error(f"User ID mismatch: Authenticated ID ({auth_user.user.id}) != Insert ID ({user_id})")
                return

            if "access_token" in st.session_state and "refresh_token" in st.session_state:
                supabase.auth.set_session(st.session_state["access_token"], st.session_state["refresh_token"])

            supabase.storage.from_("plant-images").upload(
                file_name,
                image_data.getvalue(),
                file_options={"content-type": "image/png"}
            )
            supabase.table("classification_history").insert(insert_data).execute()
            st.success("History saved successfully")
        except Exception as e:
            if "refresh_token" in st.session_state and ("403" in str(e) or "42501" in str(e)):
                try:
                    st.write("Debug: Attempting token refresh...")
                    new_session = supabase.auth.refresh_session(st.session_state["refresh_token"])
                    st.write(f"Debug: New session user ID: {new_session.user.id}")
                    st.session_state["access_token"] = new_session.session.access_token
                    st.session_state["refresh_token"] = new_session.session.refresh_token
                    supabase.auth.set_session(new_session.session.access_token, new_session.session.refresh_token)
                    auth_user = supabase.auth.get_user()
                    if auth_user.user.id != user_id:
                        st.error(f"Post-refresh user ID mismatch: Authenticated ID ({auth_user.user.id}) != Insert ID ({user_id})")
                        return
                    supabase.table("classification_history").insert(insert_data).execute()
                    st.success("History saved after token refresh")
                except Exception as refresh_error:
                    st.error(f"Token refresh failed: {refresh_error}. Please log in again.")
            else:
                st.error(f"Error saving to history: {e}")

def get_user_history(user_id):
    if user_id:
        try:
            if "access_token" in st.session_state and "refresh_token" in st.session_state:
                supabase.auth.set_session(st.session_state["access_token"], st.session_state["refresh_token"])
            response = supabase.table("classification_history")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("timestamp", desc=True)\
                .execute()
            history = response.data
            for item in history:
                image_url = supabase.storage.from_("plant-images").get_public_url(item["image_path"])
                item["image_url"] = image_url
            return history
        except Exception as e:
            st.error(f"Error fetching history: {e}")
            return []
    return []

# ------------------------- FETCH DATA FROM SUPABASE -------------------------
def get_disease_info(disease_name):
    try:
        if "access_token" in st.session_state and "refresh_token" in st.session_state:
            supabase.auth.set_session(st.session_state["access_token"], st.session_state["refresh_token"])
        
        response = supabase.table("disease")\
            .select(
                "disease_name, description, "
                "symptoms(symptom_description), "
                "treatment_suggestions(treatment_description), "
                "fertiliser(fertiliser_name, type)"
            )\
            .eq("disease_name", disease_name)\
            .execute()

        if response.data:
            disease_info = response.data[0]
            return {
                "Disease Name": disease_info["disease_name"],
                "Description": disease_info["description"],
                "Symptoms": ", ".join(
                    [s["symptom_description"] for s in disease_info.get("symptoms", [])]
                ) if disease_info.get("symptoms") else "No symptoms listed",
                "Treatment": (
                    disease_info["treatment_suggestions"][0]["treatment_description"]
                    if disease_info.get("treatment_suggestions") else "No specific treatment available"
                ),
                "Fertiliser": (
                    f"{disease_info['fertiliser'][0]['fertiliser_name']} ({disease_info['fertiliser'][0]['type']})"
                    if disease_info.get("fertiliser") else "No recommended fertiliser"
                )
            }
        else:
            return None
    except Exception as e:
        st.error(f"Database fetch error: {e}")
        return None

# ------------------------- CHATBOT FUNCTION WITH GOOGLE API -------------------------
def agriculture_chatbot(user_input):
    try:
        prompt = f"As an agricultural expert, answer this: {user_input}"
        response = chatbot_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Chatbot error: {e}")
        return "Sorry, I couldn’t process your request. Please try again!"

# ------------------------- LOGOUT FUNCTION -------------------------
def logout():
    supabase.auth.sign_out()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Logged out successfully!")
    st.experimental_rerun()

# ------------------------- MAIN PAGE -------------------------
def main_page():
    global supabase
    if "user" not in st.session_state:
        login_page()
        return

    if "access_token" in st.session_state and "refresh_token" in st.session_state:
        try:
            current_session = supabase.auth.get_session()
            if not current_session:
                supabase.auth.set_session(st.session_state["access_token"], st.session_state["refresh_token"])
        except Exception as e:
            st.error(f"Error setting session: {e}")
            if "refresh_token" in st.session_state:
                try:
                    new_session = supabase.auth.refresh_session(st.session_state["refresh_token"])
                    st.session_state["access_token"] = new_session.session.access_token
                    st.session_state["refresh_token"] = new_session.session.refresh_token
                    supabase.auth.set_session(new_session.session.access_token, new_session.session.refresh_token)
                except Exception as refresh_error:
                    st.error(f"Token refresh failed: {refresh_error}. Please log in again.")
                    logout()

    image_path = os.path.join(os.path.dirname(__file__), "images", "background_image.jpg")
    with open(image_path, "rb") as img_file:
        background_image_base64 = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{background_image_base64});
            background-size: cover;
            background-position: center;
            height: 100vh;
        }}
        .sidebar .sidebar-content {{
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        menu_options = ["Home", "History", "Chatbot", "About"]
        menu = st.selectbox("Menu", menu_options, key="main_menu")
        
        if st.button("Logout"):
            logout()

    if menu == "Home":
        st.markdown("<h1 style='color: white;'>Plant Disease Classifier</h1>", unsafe_allow_html=True)
        if "user" in st.session_state and st.session_state["user"] != "guest":
            auth_user = supabase.auth.get_user()
            st.write(f"Logged in as user_id: {auth_user.user.id if auth_user else 'Not authenticated'}")

        uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            col1, col2 = st.columns(2)

            with col1:
                resized_img = image.resize((150, 150))
                img_base64 = base64.b64encode(uploaded_image.getvalue()).decode()
                st.markdown(
                    f"""
                    <div style="border: 5px solid white; display: inline-block; padding: 0;">
                        <img src="data:image/png;base64,{img_base64}" style="width: 150px; height: 150px;" />
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                if st.button('Classify'):
                    with st.spinner('Classifying...'):
                        prediction = predict_image_class(disease_model, uploaded_image, class_indices)
                        st.markdown(f"<p style='color: blue; background-color: white; padding: 10px; border-radius: 5px;'>Prediction: {prediction}</p>", unsafe_allow_html=True)

                        if st.session_state["user"] != "guest":
                            auth_user = supabase.auth.get_user()
                            if auth_user:
                                user_id = auth_user.user.id
                                save_to_history(user_id, uploaded_image, prediction)
                            else:
                                st.error("Not authenticated. Please log in again.")

                        # Handle Healthy prediction separately
                        if prediction.lower() == "healthy":
                            st.markdown("<p style='color: green; background-color: white; padding: 10px; border-radius: 5px;'>This plant appears to be healthy!</p>", unsafe_allow_html=True)
                        else:
                            disease_info = get_disease_info(prediction)
                            if disease_info:
                                st.subheader("📌 Disease Details")
                                st.write(f"**🦠 Disease:** {disease_info['Disease Name']}")
                                st.write(f"**📚 Description:** {disease_info['Description']}")
                                st.write(f"**📚 Symptoms:** {disease_info['Symptoms']}")
                                st.write(f"**💊 Treatment:** {disease_info['Treatment']}")
                                st.write(f"**🌿 Recommended Fertiliser:** {disease_info['Fertiliser']}")
                            else:
                                st.warning("⚠️ No additional information found in the database.")

    elif menu == "History":
        if st.session_state["user"] == "guest":
            st.warning("History is only available for logged-in users. Please log in to view your classification history.")
        else:
            st.markdown("<h1 style='color: white;'>Classification History</h1>", unsafe_allow_html=True)
            auth_user = supabase.auth.get_user()
            if auth_user:
                history = get_user_history(auth_user.user.id)
                if not history:
                    st.info("No classification history yet.")
                else:
                    for item in history:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(item["image_url"], width=100)
                        with col2:
                            st.write(f"**Prediction:** {item['prediction']}")
                            st.write(f"**Date:** {item['timestamp']}")
                        st.markdown("---")
            else:
                st.error("Not authenticated. Please log in again.")

    elif menu == "Chatbot":
        st.markdown("<h1 style='color: white;'>Agriculture Chatbot</h1>", unsafe_allow_html=True)
        st.write("Ask me anything about agriculture, plant diseases, or farming tips!")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_input = st.text_input("You:", key="chat_input")
        if st.button("Send"):
            if user_input:
                with st.spinner("Thinking..."):
                    response = agriculture_chatbot(user_input)
                st.session_state["chat_history"].append({"user": user_input, "bot": response})

        for chat in st.session_state["chat_history"][::-1]:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Bot:** {chat['bot']}")
            st.markdown("---")

    elif menu == "About":
        st.markdown("<h1>About the Plant Disease Classifier</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            This application identifies plant diseases from images and includes an AI-powered agriculture chatbot.
            - Uses machine learning (CNN model)
            - Fetches disease info from the Supabase database
            - Chatbot powered by Google Generative AI
            - Developed using Streamlit, TensorFlow, and Supabase

            ### Developed by:
            - **Nadhil Farzeen**
            - **Shifnal Shyju P**
            - **Tristin Titus**
            - **Nidhin Joy**
            """
        )

# ------------------------- RUN APP -------------------------
if __name__ == "__main__":
    main_page()
