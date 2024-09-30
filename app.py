import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Page configuration
st.set_page_config(page_title="Sentiment Analysis", page_icon="😊", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .result {
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .positive {
        background-color: #b2f2bb;
        color: #0c7c59;
    }
    .negative {
        background-color: #ffc9c9;
        color: #a50e0e;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Sentiment Analysis")
st.markdown("Analyze the sentiment of your text using fine-tuned BERT on imdb dataset")
st.markdown("[my huggingface repo](https://huggingface.co/kgpian/bert-sentiment-imdb)")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("kgpian/bert-sentiment-imdb")
    tokenizer = AutoTokenizer.from_pretrained("kgpian/bert-sentiment-imdb")
    return model, tokenizer

model, tokenizer = load_model()

# User input
st.markdown("This model is fine tuned for 1 epoch only due to limited resource and high training time so this may give some incorrect results.")
input_text = st.text_area("Enter text for sentiment prediction:", height=150)

if st.button("Predict Sentiment"):
    if input_text:
        with st.spinner("Analyzing sentiment..."):
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1)
            
            sentiment = "Positive" if prediction.item() == 1 else "Negative"
            confidence = torch.softmax(outputs.logits, dim=-1)[0][prediction.item()].item()
            
            st.markdown("<p class='big-font'>Results:</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='result {'positive' if sentiment == 'Positive' else 'negative'}'>", unsafe_allow_html=True)
            st.markdown(f"<strong>Sentiment:</strong> {sentiment}", unsafe_allow_html=True)
            st.markdown(f"<strong>Confidence:</strong> {confidence:.2%}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualization
            st.markdown("<p class='big-font'>Sentiment Distribution:</p>", unsafe_allow_html=True)
            probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
            st.bar_chart({"Negative": probs[0], "Positive": probs[1]})
    else:
        st.warning("Please enter some text to analyze.")

# Add some information about the model
st.sidebar.title("About")
st.sidebar.info("This app uses a fine-tuned BERT model to predict the sentiment of text. The model was trained on the IMDB dataset and can classify text as either positive or negative.")
st.sidebar.markdown("Made with ❤️ by Sunny Kumar")