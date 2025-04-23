import streamlit as st
import yt_dlp
from pydub import AudioSegment
import whisper
import os
import google.generativeai as genai
import re
import torch

# --- Configuration ---
st.set_page_config(page_title="AI Creative Analysis", layout="wide")

# --- Gemini API Configuration ---
try:
    GEMINI_API_KEY = "AIzaSyCm72WohnOlHU18K68bNft1aFKAnw9Cw-A"
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# --- Helper Functions ---

def download_youtube_audio_yt_dlp(url):
    """Downloads audio from a YouTube URL and saves it as MP3."""
    output_file = "audio.mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'noplaylist': True,
        'nocheckcertificate': True,
    }
    try:
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except OSError as e:
                st.warning(f"Could not remove previous audio file: {e}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if os.path.exists(output_file):
            return output_file
        else:
            for f in os.listdir('.'):
                if f.startswith("audio.") and f.endswith(('.mp3', '.m4a', '.webm', '.ogg')):
                    os.rename(f, output_file)
                    return output_file
            st.error("Audio download finished, but the expected output file 'audio.mp3' was not found.")
            return None
    except yt_dlp.utils.DownloadError as e:
        if 'ffprobe and ffmpeg not found' in str(e):
            st.error("Error: FFmpeg not found. Please install FFmpeg and add it to your system's PATH.")
            st.error("Download FFmpeg from: https://ffmpeg.org/download.html or https://www.gyan.dev/ffmpeg/builds/")
        else:
            st.error(f"Error downloading video: {e}. Please check the URL and network connection.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        return None

@st.cache_resource
def load_whisper_model(model_size="large-v3"):
    """Loads the Whisper model with improved error handling."""
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using {device.upper()} for transcription")
        
        model = whisper.load_model(model_size, device=device)
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model '{model_size}': {e}")
        st.warning("Ensure you have PyTorch installed. You can install it via: pip install torch torchvision torchaudio")
        return None

@st.cache_data
def transcribe_audio(filepath, _whisper_model, language=None):
    """Transcribes the audio file using Whisper with improved settings."""
    if not filepath or not os.path.exists(filepath):
        st.error(f"Audio file not found at path: {filepath}")
        return None
    if not _whisper_model:
        st.error("Whisper model not loaded.")
        return None
    try:
        # Use improved transcription settings
        result = _whisper_model.transcribe(
            filepath,
            fp16=False,  # More stable on CPU
            language=language,  # Use specified language if provided
            verbose=True,  # Show progress
            task="transcribe",  # Force transcription mode
            temperature=0.0,  # More deterministic output
            best_of=5,  # Better quality
            beam_size=5  # Better quality
        )
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

def parse_gemini_analysis(text):
    """Parses the Gemini response text to extract score, works, and does not work."""
    score = None
    works = []
    does_not_work = []

    try:
        # Extract Score
        score_match = re.search(r"Score:\s*(\d+)\s*/\s*100", text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))

        # Extract Works
        works_match = re.search(r"Works:(.*?)(DoesNotWork:|What doesn't work:|$)", text, re.IGNORECASE | re.DOTALL)
        if works_match:
            works_text = works_match.group(1).strip()
            works = [line.strip('- ').strip() for line in works_text.split('\n') if line.strip() and line.strip().startswith('-')]

        # Extract Does Not Work
        does_not_work_match = re.search(r"(?:DoesNotWork:|What doesn't work:)(.*?)$", text, re.IGNORECASE | re.DOTALL)
        if does_not_work_match:
            does_not_work_text = does_not_work_match.group(1).strip()
            does_not_work = [line.strip('- ').strip() for line in does_not_work_text.split('\n') if line.strip() and line.strip().startswith('-')]

    except Exception as e:
        st.error(f"Error parsing Gemini analysis: {e}")
        return score, text.split('\n'), []

    if score is None and not works and not does_not_work:
        st.warning("Could not parse the Gemini response structure. Displaying raw output.")
        return None, text.split('\n'), []

    return score, works, does_not_work

def analyze_with_gemini(transcript):
    """Analyzes the transcript using the Gemini API."""
    prompt = f"""
You are an AI Creative Analyst specializing in evaluating video ad transcripts for effectiveness, particularly for a wellness audience. Analyze the following transcript:

"{transcript}"

Provide your analysis in the following format, strictly adhering to this structure:

Score: [Overall score out of 100]
Works:
- [Point 1 about what works]
- [Point 2 about what works]
...
DoesNotWork:
- [Point 1 about what doesn't work]
- [Point 2 about what doesn't work]
...

Focus your analysis on:
- Clarity of message
- Target audience resonance (assume wellness focus)
- Benefit articulation
- Call to action effectiveness (if present)
- Overall engagement potential based *only* on the text.
Ensure the output starts exactly with "Score:", followed by "Works:", and then "DoesNotWork:".
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

# --- Streamlit App UI ---
st.title("📊 AI Creative Analysis")
st.markdown("Enter a YouTube video URL to analyze its content and get an AI-powered analysis.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    
    # Model selection
    whisper_model_size = st.selectbox(
        "Select Whisper Model Size",
        ("large-v3", "medium", "small", "base", "tiny"),
        index=0,
        help="Larger models are more accurate but slower. 'large-v3' is recommended for best results."
    )
    
    # Language selection
    language = st.selectbox(
        "Select Audio Language",
        ("auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"),
        index=0,
        help="Select 'auto' for automatic language detection, or specify the language for better accuracy."
    )
    if language == "auto":
        language = None

# Main content
youtube_link = st.text_input(
    "Enter YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=...",
    key="youtube_url_input"
)

analyze_button = st.button("Analyze Video", key="analyze_button", type="primary")

if analyze_button and youtube_link:
    # Progress container
    progress_container = st.container()
    
    with progress_container:
        # Step 1: Download Audio
        st.subheader("Step 1: Downloading Audio")
        audio_file = download_youtube_audio_yt_dlp(youtube_link)
        
        if audio_file:
            st.success("✅ Audio downloaded successfully")
            
            # Step 2: Load Whisper Model
            st.subheader("Step 2: Loading Transcription Model")
            whisper_model = load_whisper_model(whisper_model_size)
            
            if whisper_model:
                st.success("✅ Model loaded successfully")
                
                # Step 3: Transcribe Audio
                st.subheader("Step 3: Transcribing Audio")
                transcript = transcribe_audio(audio_file, whisper_model, language)
                
                if transcript:
                    st.success("✅ Transcription complete")
                    
                    # Show transcript in expander
                    with st.expander("View Transcript", expanded=True):
                        st.text_area("Transcript:", transcript, height=200)
                    
                    # Step 4: Analyze with Gemini
                    st.subheader("Step 4: Analyzing Content")
                    analysis_text = analyze_with_gemini(transcript)
                    
                    if analysis_text:
                        st.success("✅ Analysis complete")
                        
                        # Parse and display results
                        score, works_list, does_not_work_list = parse_gemini_analysis(analysis_text)
                        
                        # Display Results
                        st.markdown("---")
                        st.markdown("<h2 style='text-align: center;'>Analysis Results</h2>", unsafe_allow_html=True)
                        
                        # Score display
                        if score is not None:
                            score_color = "green" if score >= 70 else ("orange" if score >= 40 else "red")
                            st.markdown(f"""
                            <div style='text-align: center; margin: 20px 0;'>
                                <div style='display: inline-block; border: 5px solid {score_color}; border-radius: 50%; padding: 20px; width: 120px; height: 120px; line-height: 80px; font-size: 2.5em; font-weight: bold;'>
                                    {score}
                                </div>
                                <div style='margin-top: 10px; font-size: 1.2em;'>Overall Score</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Analysis columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("<h3 style='color: green;'>✅ What Works</h3>", unsafe_allow_html=True)
                            if works_list:
                                for item in works_list:
                                    st.markdown(f"• {item}")
                            else:
                                st.info("No specific 'What Works' points identified.")
                        
                        with col2:
                            st.markdown("<h3 style='color: red;'>❌ What Doesn't Work</h3>", unsafe_allow_html=True)
                            if does_not_work_list:
                                for item in does_not_work_list:
                                    st.markdown(f"• {item}")
                            else:
                                st.info("No specific 'What Doesn't Work' points identified.")
                    else:
                        st.error("Analysis failed. Please try again.")
                else:
                    st.error("Transcription failed. Please try again.")
            else:
                st.error("Failed to load transcription model. Please try again.")
        else:
            st.error("Failed to download audio. Please check the URL and try again.")

elif analyze_button and not youtube_link:
    st.warning("Please enter a YouTube video URL.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    Note: Analysis is based on audio transcription and may take several minutes for longer videos.
    For best results, use videos with clear audio and select the appropriate language.
</div>
""", unsafe_allow_html=True) 