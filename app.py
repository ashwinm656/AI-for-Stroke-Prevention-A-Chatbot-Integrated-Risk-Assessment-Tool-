import streamlit as st
import pickle
import numpy as np
import requests
from hugchat import hugchat
from hugchat.login import Login
import base64

# ✅ Set page config FIRST
st.set_page_config(page_title="Stroke Risk App", layout="wide")

# ✅ Set background image
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("bg_image.jpg")  # This comes after set_page_config

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select a Page",
    ["🏠 Home", "🩺 Stroke Prediction", "📰 Stroke News", "💬 Chatbot"]
)

# Load model and scaler
model = pickle.load(open('stroke_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# News API Key
NEWS_API_KEY = "c886b1766bdc4af69e19811aef4dc9e8"

# HugChat credentials
HF_EMAIL = "amreenrafiq10@gmail.com"
HF_PASS = "v/PkBzwiJdbW@4!"
BASE_PROMPT = "Hello, I would like to ask about stroke prevention and symptoms. "
flag = 0

def query_hugchat(prompt_input):
    global flag
    sign = Login(HF_EMAIL, HF_PASS)
    cookies = sign.login()
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    if flag < 1:
        flag += 1
        prompt_input = BASE_PROMPT + prompt_input
    return str(chatbot.chat(prompt_input)).strip('`')

if page == "🏠 Home":
    st.title("Welcome to AI for Stroke Prevention")
    st.markdown("""
    This AI-powered application allows users to:
    - Predict stroke risk using health data  
    - Read latest stroke-related news  
    - Chat with a medical assistant powered by Hugging Face  

    **Developed by Ashwin Muralidharan**  
    Masters in Data Science, University of North Texas
    """)

elif page == "🩺 Stroke Prediction":
    st.header("Stroke Risk Assessment")
    with st.form("stroke_form"):
        age = st.slider("Age", 1, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_val = 1 if gender == "Male" else 0
        symptoms = {
            "chest_pain": st.checkbox("Chest Pain"),
            "high_blood_pressure": st.checkbox("High Blood Pressure"),
            "irregular_heartbeat": st.checkbox("Irregular Heartbeat"),
            "shortness_of_breath": st.checkbox("Shortness of Breath"),
            "fatigue_weakness": st.checkbox("Fatigue or Weakness"),
            "dizziness": st.checkbox("Dizziness"),
            "swelling_edema": st.checkbox("Swelling or Edema"),
            "neck_jaw_pain": st.checkbox("Neck or Jaw Pain"),
            "excessive_sweating": st.checkbox("Excessive Sweating"),
            "persistent_cough": st.checkbox("Persistent Cough"),
            "nausea_vomiting": st.checkbox("Nausea or Vomiting"),
            "chest_discomfort": st.checkbox("Chest Discomfort"),
            "cold_hands_feet": st.checkbox("Cold Hands/Feet"),
            "snoring_sleep_apnea": st.checkbox("Snoring / Sleep Apnea"),
            "anxiety_doom": st.checkbox("Anxiety / Feeling of Doom")
        }
        submitted = st.form_submit_button("🔍 Predict Stroke Risk")
        if submitted:
            age_scaled = scaler.transform([[age]])[0][0]
            input_data = [age_scaled, gender_val] + [int(symptoms[k]) for k in symptoms]
            prediction = model.predict([input_data])[0]
            label = "⚠️ At Risk of Stroke" if prediction == 1 else "✅ Not at Risk"
            st.success(f"Prediction: {label}")

elif page == "📰 Stroke News":
    st.header("Latest Stroke News")
    if st.button("🔄 Refresh News"):
        try:
            response = requests.get(f"https://newsapi.org/v2/everything?q=stroke&apiKey={NEWS_API_KEY}")
            articles = response.json().get("articles", [])[:5]
            for article in articles:
                st.subheader(article['title'])
                st.write(article['description'])
                st.markdown(f"[Read more]({article['url']})")
                st.markdown("---")
        except:
            st.error("Failed to load news.")

elif page == "💬 Chatbot":
    st.header("Chat with Stroke Assistant 🧠")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How may I help you regarding stroke?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Type your question about stroke here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = query_hugchat(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error("Something went wrong with the chatbot.")
                    st.error(str(e))
