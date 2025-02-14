import streamlit as st
import whisper
import tempfile
import os



# Set page config
st.set_page_config(page_title="Audio Wizard", page_icon="🎙️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .st-emotion-cache-1v0mbdj {
        width: 100%;
    }
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
    .transcribe-button:hover {
        background-color: #3a3f9e;
    }
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
st.markdown("Transform your audio into text with the power of AI! Upload your file, hit transcribe, and watch the magic happen.")

# File uploader
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
st.markdown("</div>", unsafe_allow_html=True)

model_name = st.selectbox(
    "Choose a Whisper model:",
    ["tiny", "base", "small", "medium", "large"],
    index=1
)

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Transcribe button
    if st.button("🪄 Transcribe Audio", key="transcribe_button", help="Click to start transcription"):
        with st.spinner("🔮 Casting transcription spell..."):
            # Load the Whisper model
            model = whisper.load_model(model_name)

            # Transcribe the audio
            result = model.transcribe(tmp_file_path,verbose=1)

            # Display the transcribed text
            st.markdown("<div class='transcription-box'>", unsafe_allow_html=True)
            st.subheader("✨ Transcription Result:")
            st.write(result["text"])
            st.markdown("</div>", unsafe_allow_html=True)

    # Clean up the temporary file
    os.unlink(tmp_file_path)

else:
    st.info("👆 Please upload an audio file to begin the transcription journey.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Audio Wizard Team")