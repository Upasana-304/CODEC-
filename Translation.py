# app.py
# Speech-to-Text Transcription Tool using Streamlit
# Requirements:
# pip install streamlit SpeechRecognition pydub

import streamlit as st
import speech_recognition as sr
import tempfile
import os

# Optional audio conversion support
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# PAGE CONFIG 
st.set_page_config(
    page_title="Speech-to-Text Transcription Tool",
    page_icon="🎤",
    layout="centered"
)

#  TITLE 
st.title("🎤 Speech-to-Text Transcription Tool")
st.write("Upload an audio file and convert speech into text.")

# FUNCTIONS 
def convert_to_wav(input_file_path):
    if AudioSegment is None:
        return input_file_path

    file_extension = input_file_path.split(".")[-1].lower()

    if file_extension == "wav":
        return input_file_path

    sound = AudioSegment.from_file(input_file_path)

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sound.export(temp_wav.name, format="wav")

    return temp_wav.name


def transcribe_audio(file_path):
    recognizer = sr.Recognizer()

    try:
        wav_file = convert_to_wav(file_path)

        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return text

    except sr.UnknownValueError:
        return "Could not understand the audio."

    except sr.RequestError:
        return "Speech Recognition service error."

    except Exception as e:
        return f"Error: {str(e)}"


# FILE UPLOAD 
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "m4a", "ogg", "flac"]
)

# MAIN PROCESS 
if uploaded_file is not None:

    st.audio(uploaded_file, format="audio/wav")

    # Save uploaded file temporarily
    file_extension = uploaded_file.name.split(".")[-1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    if st.button("Convert to Text"):

        with st.spinner("Transcribing audio..."):
            result = transcribe_audio(temp_file.name)

        st.subheader("Transcribed Text:")
        st.success(result)

        st.download_button(
            label="Download Text File",
            data=result,
            file_name="transcription.txt",
            mime="text/plain"
        )

    # Delete temp file after use
    try:
        os.unlink(temp_file.name)
    except:
        pass

#  FOOTER 
st.markdown("---")
st.write("Developed with Python + Streamlit")