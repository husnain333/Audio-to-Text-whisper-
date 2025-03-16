import streamlit as st
import whisper
import tempfile
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from gtts import gTTS

# Set page config
st.set_page_config(page_title="Audio Wizard", page_icon="🎙️", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stApp { max-width: 800px; margin: 0 auto; }
    .st-emotion-cache-1v0mbdj { width: 100%; }
    .upload-box {
        border: 2px dashed #4e54c8;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #ffffff;
    }
    .transcribe-button {
        background-color: #4e54c8;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .transcribe-button:hover { background-color: #3a3f9e; }
    .transcription-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎙️ Audio Wizard")
st.markdown("Transform your audio into text and then into a summarized voice note with the power of AI!")

# File uploader
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
st.markdown("</div>", unsafe_allow_html=True)

# Whisper model selection
model_name = st.selectbox(
    "Choose a Whisper model:",
    ["tiny", "base", "small", "medium", "large"],
    index=1
)

# Load T5-base model and tokenizer (cache it to avoid reloading)
@st.cache_resource(show_spinner=False)
def load_summarizer():
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    return tokenizer, model

summarizer_tokenizer, summarizer_model = load_summarizer()

# Function to summarize text using T5 with dynamic length
def summarize_text(text, min_len=30, max_len=150):
    input_text = "summarize: " + text.strip().replace("\n", " ")
    
    # Tokenize the input
    inputs = summarizer_tokenizer.encode(
        input_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    )
    
    # Generate summary dynamically
    summary_ids = summarizer_model.generate(
        inputs,
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to convert text to speech and save as an MP3
def text_to_speech(summary_text, lang='en'):
    tts = gTTS(text=summary_text, lang=lang)
    tts_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(tts_file.name)
    return tts_file.name

# If audio file is uploaded
if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')

    # Create a temporary file to save uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Length control sliders (for dynamic summary length)
    st.markdown("### 📏 Customize Your Summary Length")
    max_length = st.slider(
        "Select the maximum length of the summary (tokens):",
        min_value=30, max_value=512, value=150, step=10
    )

    min_length = st.slider(
        "Select the minimum length of the summary (tokens):",
        min_value=10, max_value=max_length - 10, value=30, step=5
    )

    # Transcribe button
    if st.button("🪄 Transcribe Audio", key="transcribe_button", help="Click to start transcription"):
        with st.spinner("🔮 Casting transcription spell..."):
            # Load the Whisper model
            model = whisper.load_model(model_name)

            # Transcribe the audio
            result = model.transcribe(tmp_file_path, verbose=1)

            # Display the transcribed text
            st.markdown("<div class='transcription-box'>", unsafe_allow_html=True)
            st.subheader("✨ Transcription Result:")
            st.write(result["text"])
            st.markdown("</div>", unsafe_allow_html=True)

            # Summarize the transcription with user-defined lengths
            with st.spinner("📝 Summarizing the transcription..."):
                summary = summarize_text(result["text"], min_len=min_length, max_len=max_length)

            # Display the summary
            st.markdown("<div class='transcription-box'>", unsafe_allow_html=True)
            st.subheader("📚 Summary:")
            st.write(summary)
            st.markdown("</div>", unsafe_allow_html=True)

            # Convert the summary to speech using gTTS
            with st.spinner("🎧 Converting summary to speech..."):
                tts_file_path = text_to_speech(summary)

            # Play the generated speech
            st.markdown("<div class='transcription-box'>", unsafe_allow_html=True)
            st.subheader("🔊 Listen to the Summary:")
            audio_file = open(tts_file_path, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3')

            # Download button for the generated summary speech
            st.download_button(
                label="💾 Download Summary Audio",
                data=audio_bytes,
                file_name="summary_audio.mp3",
                mime="audio/mp3"
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # Clean up the temporary file after processing
    os.unlink(tmp_file_path)

else:
    st.info("👆 Please upload an audio file to begin the transcription journey.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Audio Wizard Team")
