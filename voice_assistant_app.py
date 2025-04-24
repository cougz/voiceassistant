import os
import io
import numpy as np
import tempfile
import asyncio
import chainlit as cl
from chainlit.input_widget import Select, Switch
from openai import OpenAI
import riva.client
from dotenv import load_dotenv
import srt
import datetime
from typing import List, Tuple, Dict, Optional
import wave
import re
import requests

# Access environment variables from .env file
load_dotenv()
ai_endpoint_token = os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
llm_endpoint = os.getenv("LLM_AI_ENDPOINT") # Get LLM endpoint from .env

# Check if environment variables are loaded
if not ai_endpoint_token:
    raise ValueError("OVH_AI_ENDPOINTS_ACCESS_TOKEN not found in .env file. Please create a .env file with this variable.")
if not llm_endpoint:
    raise ValueError("LLM_AI_ENDPOINT not found in .env file. Please create a .env file with this variable.")


# --- Language Definitions ---

# ASR Languages: Based on known/provided OVHcloud Endpoints
# Add more entries here if you find OVHcloud ASR endpoints for other languages.
ASR_LANGUAGES = {
    "English (US)": {
        "endpoint": "nvr-asr-en-us.endpoints-grpc.kepler.ai.cloud.ovh.net:443",
        "language_code": "en-US"
    },
    # Example Placeholder (replace with actual endpoint if available):
    # "Spanish (ES)": {
    #     "endpoint": "nvr-asr-es-es.endpoints-grpc.kepler.ai.cloud.ovh.net:443", # <-- FIND ACTUAL SPANISH ASR ENDPOINT
    #     "language_code": "es-ES"
    # },
}

# TTS Languages: Based on the voices listed in the OVHcloud nvr-tts-en-us endpoint documentation
# Using the nvr-tts-en-us endpoint URI for all as per documentation examples.
# Verify with OVH if language-specific TTS endpoints should be used.
TTS_ENDPOINT_URI = "nvr-tts-en-us.endpoints-grpc.kepler.ai.cloud.ovh.net:443"

TTS_LANGUAGES = {
    # --- English (US) ---
    "English (US) Female": {"endpoint": TTS_ENDPOINT_URI, "language_code": "en-US", "voice_name": "English-US.Female-1"},
    "English (US) Male": {"endpoint": TTS_ENDPOINT_URI, "language_code": "en-US", "voice_name": "English-US.Male-1"},
    "English (US) Female Calm": {"endpoint": TTS_ENDPOINT_URI, "language_code": "en-US", "voice_name": "English-US.Female-Calm"},
    "English (US) Female Neutral": {"endpoint": TTS_ENDPOINT_URI, "language_code": "en-US", "voice_name": "English-US.Female-Neutral"},
    "English (US) Female Happy": {"endpoint": TTS_ENDPOINT_URI, "language_code": "en-US", "voice_name": "English-US.Female-Happy"},
    "English (US) Female Angry": {"endpoint": TTS_ENDPOINT_URI, "language_code": "en-US", "voice_name": "English-US.Female-Angry"},
    "English (US) Female Fearful": {"endpoint": TTS_ENDPOINT_URI, "language_code": "en-US", "voice_name": "English-US.Female-Fearful"},
    "English (US) Female Sad": {"endpoint": TTS_ENDPOINT_URI, "language_code": "en-US", "voice_name": "English-US.Female-Sad"},
    "English (US) Male Calm": {"endpoint": TTS_ENDPOINT_URI, "language_code": "en-US", "voice_name": "English-US.Male-Calm"},
    "English (US) Male Neutral": {"endpoint": TTS_ENDPOINT_URI, "language_code": "en-US", "voice_name": "English-US.Male-Neutral"},
    "English (US) Male Happy": {"endpoint": TTS_ENDPOINT_URI, "language_code": "en-US", "voice_name": "English-US.Male-Happy"},
    "English (US) Male Angry": {"endpoint": TTS_ENDPOINT_URI, "language_code": "en-US", "voice_name": "English-US.Male-Angry"},

    # --- Spanish (ES) ---
    "Spanish (ES) Female": {"endpoint": TTS_ENDPOINT_URI, "language_code": "es-ES", "voice_name": "Spanish-ES-Female-1"},
    "Spanish (ES) Male": {"endpoint": TTS_ENDPOINT_URI, "language_code": "es-ES", "voice_name": "Spanish-ES-Male-1"},

    # --- German (DE) ---
    "German (DE) Male": {"endpoint": TTS_ENDPOINT_URI, "language_code": "de-DE", "voice_name": "German-DE-Male-1"},

    # --- Italian (IT) ---
    "Italian (IT) Female": {"endpoint": TTS_ENDPOINT_URI, "language_code": "it-IT", "voice_name": "Italian-IT-Female-1"},
    "Italian (IT) Male": {"endpoint": TTS_ENDPOINT_URI, "language_code": "it-IT", "voice_name": "Italian-IT-Male-1"},

    # --- Mandarin (CN) ---
    "Mandarin (CN) Female": {"endpoint": TTS_ENDPOINT_URI, "language_code": "zh-CN", "voice_name": "Mandarin-CN.Female-1"},
    "Mandarin (CN) Male": {"endpoint": TTS_ENDPOINT_URI, "language_code": "zh-CN", "voice_name": "Mandarin-CN.Male-1"},
    "Mandarin (CN) Female Calm": {"endpoint": TTS_ENDPOINT_URI, "language_code": "zh-CN", "voice_name": "Mandarin-CN.Female-Calm"},
    "Mandarin (CN) Female Neutral": {"endpoint": TTS_ENDPOINT_URI, "language_code": "zh-CN", "voice_name": "Mandarin-CN.Female-Neutral"},
    "Mandarin (CN) Male Happy": {"endpoint": TTS_ENDPOINT_URI, "language_code": "zh-CN", "voice_name": "Mandarin-CN.Male-Happy"},
    "Mandarin (CN) Male Fearful": {"endpoint": TTS_ENDPOINT_URI, "language_code": "zh-CN", "voice_name": "Mandarin-CN.Male-Fearful"},
    "Mandarin (CN) Male Sad": {"endpoint": TTS_ENDPOINT_URI, "language_code": "zh-CN", "voice_name": "Mandarin-CN.Male-Sad"},
    "Mandarin (CN) Male Calm": {"endpoint": TTS_ENDPOINT_URI, "language_code": "zh-CN", "voice_name": "Mandarin-CN.Male-Calm"},
    "Mandarin (CN) Male Neutral": {"endpoint": TTS_ENDPOINT_URI, "language_code": "zh-CN", "voice_name": "Mandarin-CN.Male-Neutral"},
    "Mandarin (CN) Male Angry": {"endpoint": TTS_ENDPOINT_URI, "language_code": "zh-CN", "voice_name": "Mandarin-CN.Male-Angry"},
}


# --- Global Settings & Options ---

DEFAULT_ASR_LANGUAGE = "English (US)" # Default ASR still English
DEFAULT_TTS_VOICE = "English (US) Female" # Default TTS voice
DEFAULT_GENERATE_SUBTITLES = False
DEFAULT_TTS_ENABLED = True
DEFAULT_SAMPLE_RATE = "48000 Hz" # Higher default sample rate
DEFAULT_PAUSE_DURATION = "Medium (0.3s)"

SAMPLE_RATES = {
    "16000 Hz": 16000,
    "22050 Hz": 22050,
    "44100 Hz": 44100,
    "48000 Hz": 48000 # Added 48k based on Riva capabilities
}

PAUSE_DURATIONS = {
    "Short (0.2s)": 0.2,
    "Medium (0.3s)": 0.3,
    "Long (0.5s)": 0.5
}

# --- Chainlit UI Setup ---

@cl.on_chat_start
async def start():
    # Initialize conversation
    await cl.Message(content="Hello! I'm your AI Voice Assistant. Select your preferred languages and voice in settings, then speak or type!", author="AI Assistant").send()

    # Find initial index for defaults
    try:
        tts_initial_index = list(TTS_LANGUAGES.keys()).index(DEFAULT_TTS_VOICE)
    except ValueError:
        tts_initial_index = 0 # Fallback if default not found

    try:
        asr_initial_index = list(ASR_LANGUAGES.keys()).index(DEFAULT_ASR_LANGUAGE)
    except ValueError:
        asr_initial_index = 0 # Fallback

    try:
        sr_initial_index = list(SAMPLE_RATES.keys()).index(DEFAULT_SAMPLE_RATE)
    except ValueError:
        sr_initial_index = len(SAMPLE_RATES) - 1 # Fallback to highest rate

    try:
        pause_initial_index = list(PAUSE_DURATIONS.keys()).index(DEFAULT_PAUSE_DURATION)
    except ValueError:
        pause_initial_index = 1 # Fallback to medium

    # Setup settings with ChatSettings
    settings = await cl.ChatSettings(
        [
            Switch(id="tts_enabled", label="Enable Text-to-Speech", initial=DEFAULT_TTS_ENABLED),
            Select(id="tts_voice", label="Text-to-Speech Voice", values=list(TTS_LANGUAGES.keys()), initial_index=tts_initial_index),
            Select(id="sample_rate", label="Audio Quality (Sample Rate)", values=list(SAMPLE_RATES.keys()), initial_index=sr_initial_index),
            Select(id="pause_duration", label="Pause Between Sentences", values=list(PAUSE_DURATIONS.keys()), initial_index=pause_initial_index),
            Select(id="asr_language", label="Speech Recognition Language", values=list(ASR_LANGUAGES.keys()), initial_index=asr_initial_index, description="Requires corresponding OVHcloud ASR endpoint."),
            Switch(id="generate_subtitles", label="Generate Subtitles", initial=DEFAULT_GENERATE_SUBTITLES)
        ]
    ).send()

    # Store initial settings in user session
    cl.user_session.set("settings", {
        "tts_enabled": DEFAULT_TTS_ENABLED,
        "tts_voice": DEFAULT_TTS_VOICE,
        "asr_language": DEFAULT_ASR_LANGUAGE,
        "generate_subtitles": DEFAULT_GENERATE_SUBTITLES,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "pause_duration": DEFAULT_PAUSE_DURATION
    })

    cl.user_session.set("audio_buffer", io.BytesIO())
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("transcription_buffer", "")
    cl.user_session.set("processing_message_id", None)
    cl.user_session.set("has_spoken", False) # Flag to track if the user has started speaking

@cl.on_audio_start
async def on_audio_start():
    print("@@@ ON AUDIO START CALLED @@@") # Added logging
    cl.user_session.set("audio_buffer", io.BytesIO())
    cl.user_session.set("is_speaking", True)
    cl.user_session.set("transcription_buffer", "")
    cl.user_session.set("processing_message_id", None)
    cl.user_session.set("has_spoken", True) # User has started speaking
    processing_msg = cl.Message(content="Listening...", author="System")
    await processing_msg.send()
    cl.user_session.set("processing_message_id", processing_msg.id)

# Handle settings update
@cl.on_settings_update
async def on_settings_update(settings: dict):
    cl.user_session.set("settings", settings)
    tts_status = "enabled" if settings.get("tts_enabled") else "disabled"
    subtitle_status = "enabled" if settings.get("generate_subtitles") else "disabled"
    settings_update = (
        f"Settings updated:\n"
        f"- TTS: {tts_status}, Voice: {settings.get('tts_voice')}\n"
        f"- Audio Quality: {settings.get('sample_rate')}\n"
        f"- Pause Duration: {settings.get('pause_duration')}\n"
        f"- Speech Recognition: {settings.get('asr_language')}\n"
        f"- Subtitles: {subtitle_status}"
    )
    await cl.Message(content=settings_update, author="System").send()


# --- Core AI Functions ---

# Automatic speech recognition - question transcription
async def asr_transcription(audio_data, language_option=DEFAULT_ASR_LANGUAGE):
    if language_option not in ASR_LANGUAGES:
        print(f"Warning: ASR language '{language_option}' not configured. Falling back to '{DEFAULT_ASR_LANGUAGE}'.")
        language_option = DEFAULT_ASR_LANGUAGE

    language_info = ASR_LANGUAGES[language_option]
    print(f"Starting ASR: Lang='{language_option}', Endpoint='{language_info['endpoint']}'")

    try:
        auth = riva.client.Auth(
            uri=language_info["endpoint"],
            use_ssl=True,
            metadata_args=[["authorization", f"bearer {ai_endpoint_token}"]]
        )
        asr_service = riva.client.ASRService(auth)
        asr_config = riva.client.RecognitionConfig(
            language_code=language_info["language_code"],
            max_alternatives=1,
            enable_automatic_punctuation=True,
            audio_channel_count=1,
        )
        response = asr_service.offline_recognize(audio_data, asr_config)

        if response.results and response.results[0].alternatives:
            transcript = response.results[0].alternatives[0].transcript
            print(f"ASR Success: '{transcript[:100]}...'")
            return transcript
        else:
            print("ASR Warning: No transcription result.")
            return ""
    except Exception as e:
        print(f"ASR Error: {e}")
        return "[ASR Error]"

# Text to speech - answer synthesis
async def tts_synthesis(text, voice_option=DEFAULT_TTS_VOICE, sample_rate_option=DEFAULT_SAMPLE_RATE, pause_duration_option=DEFAULT_PAUSE_DURATION):
    if not text:
        print("TTS Warning: No text provided for synthesis.")
        return np.array([], dtype=np.int16), SAMPLE_RATES[sample_rate_option]

    if voice_option not in TTS_LANGUAGES:
        print(f"Warning: TTS voice '{voice_option}' not configured. Falling back to '{DEFAULT_TTS_VOICE}'.")
        voice_option = DEFAULT_TTS_VOICE

    voice_info = TTS_LANGUAGES[voice_option]
    sample_rate_hz = SAMPLE_RATES.get(sample_rate_option, 44100) # Default to 44.1k if setting invalid
    pause_duration = PAUSE_DURATIONS.get(pause_duration_option, 0.3)

    print(f"Starting TTS: Voice='{voice_option}', Lang='{voice_info['language_code']}', Rate='{sample_rate_hz}Hz', Endpoint='{voice_info['endpoint']}'")
    print(f"TTS Input Text: '{text[:50]}...'")

    # Chunking logic
    max_chars_per_chunk = 500
    chunks = []
    if len(text) > max_chars_per_chunk:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        for sentence in sentences:
            # Handle sentences longer than max chunk size
            while len(sentence) > max_chars_per_chunk:
                split_point = sentence[:max_chars_per_chunk].rfind(' ')
                if split_point == -1: split_point = max_chars_per_chunk
                part = sentence[:split_point].strip()
                if part:
                    if current_chunk: chunks.append(current_chunk.strip())
                    chunks.append(part)
                    current_chunk = ""
                sentence = sentence[split_point:].strip()

            # Add sentence to current chunk if space allows
            if sentence:
                if not current_chunk:
                    current_chunk = sentence
                elif len(current_chunk) + len(sentence) + 1 <= max_chars_per_chunk:
                    current_chunk += " " + sentence
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence

        if current_chunk: chunks.append(current_chunk.strip())
    else:
        chunks = [text]

    print(f"Split into {len(chunks)} chunks for TTS.")
    all_audio_samples = np.array([], dtype=np.int16)

    # Process each chunk
    for i, chunk in enumerate(chunks):
        if not chunk: continue
        print(f"Processing chunk {i+1}/{len(chunks)}: '{chunk[:30]}...'")
        try:
            auth = riva.client.Auth(
                uri=voice_info["endpoint"],
                use_ssl=True,
                metadata_args=[["authorization", f"bearer {ai_endpoint_token}"]]
            )
            tts_service = riva.client.SpeechSynthesisService(auth)
            req = {
                "language_code": voice_info["language_code"],
                "encoding": riva.client.AudioEncoding.LINEAR_PCM,
                "sample_rate_hz": sample_rate_hz,
                "voice_name": voice_info["voice_name"],
                "text": chunk
            }
            response = tts_service.synthesize(**req)
            chunk_samples = np.frombuffer(response.audio, dtype=np.int16)
            all_audio_samples = np.concatenate([all_audio_samples, chunk_samples])

            # Add pause between chunks
            if i < len(chunks) - 1:
                pause_samples = np.zeros(int(sample_rate_hz * pause_duration), dtype=np.int16)
                all_audio_samples = np.concatenate([all_audio_samples, pause_samples])

            await asyncio.sleep(0.1) # Small delay between requests
        except Exception as e:
            print(f"TTS Error on chunk {i+1}: {e}")
            # Optionally add a silent gap or skip chunk on error

    if len(all_audio_samples) == 0:
        print("TTS Warning: Synthesis resulted in zero audio samples.")
        # Return empty array, let caller handle it
        return np.array([], dtype=np.int16), sample_rate_hz

    print(f"TTS Synthesis successful. Total samples: {len(all_audio_samples)}")
    return all_audio_samples, sample_rate_hz


# Generate SRT subtitles
def generate_srt(text):
    """Generate SRT subtitle file from text using simple timing estimation"""
    if not text: return ""
    print("Generating SRT subtitles...")
    words = text.split()
    WPM = 150 # Estimated words per minute
    words_per_second = WPM / 60.0
    words_per_subtitle = 8 # Max words per line

    subtitles = []
    start_time_seconds = 0.0
    subtitle_index = 1

    for i in range(0, len(words), words_per_subtitle):
        chunk_words = words[i:i+words_per_subtitle]
        subtitle_text = " ".join(chunk_words)
        chunk_duration_seconds = max(1.0, len(chunk_words) / words_per_second) # Min 1 sec duration

        start_td = datetime.timedelta(seconds=start_time_seconds)
        end_td = start_td + datetime.timedelta(seconds=chunk_duration_seconds)

        subtitles.append(srt.Subtitle(index=subtitle_index, start=start_td, end=end_td, content=subtitle_text))
        subtitle_index += 1
        start_time_seconds += chunk_duration_seconds

    srt_content = srt.compose(subtitles)
    print(f"Generated {len(subtitles)} subtitle entries.")
    return srt_content

# --- New function to handle audio chunks ---
@cl.on_audio_chunk
async def on_audio_chunk(chunk: bytes):
    settings = cl.user_session.get("settings", {})
    asr_language = settings.get("asr_language", DEFAULT_ASR_LANGUAGE)
    sample_rate_str = settings.get("sample_rate", DEFAULT_SAMPLE_RATE)
    sample_rate_hz = SAMPLE_RATES.get(sample_rate_str, 48000)

    audio_buffer = cl.user_session.get("audio_buffer")
    is_speaking = cl.user_session.get("is_speaking")
    transcription_buffer = cl.user_session.get("transcription_buffer")
    processing_message_id = cl.user_session.get("processing_message_id")
    has_spoken = cl.user_session.get("has_spoken")

    if not has_spoken:
        return # Ignore chunks until on_audio_start is called

    audio_buffer.write(chunk)
    cl.user_session.set("audio_buffer", audio_buffer)
    cl.user_session.set("is_speaking", True)

    if not processing_message_id:
        processing_msg = cl.Message(content="Listening...", author="System")
        await processing_msg.send()
        cl.user_session.set("processing_message_id", processing_msg.id)

    # Voice Activity Detection (Simple: check for silence)
    # This is a very basic implementation and might need more sophisticated VAD
    audio_array = np.frombuffer(chunk, dtype=np.int16)
    if np.abs(audio_array).mean() < 10: # Threshold for silence (adjust as needed)
        if is_speaking:
            cl.user_session.set("is_speaking", False)
            if transcription_buffer:
                if processing_message_id:
                    processing_message = cl.Message(content=f"Heard: \"{transcription_buffer.strip()}\"", author="System", disable_human_feedback=True, id=processing_message_id)
                    await processing_message.update()

                # Simulate a message being sent by the user with the transcribed text
                await on_message(cl.Message(content=transcription_buffer.strip(), author="User"))

                # Reset the transcription buffer and processing message ID
                cl.user_session.set("transcription_buffer", "")
                cl.user_session.set("processing_message_id", None)
                cl.user_session.set("has_spoken", False)
            elif processing_message_id:
                processing_message = cl.Message(content="No speech detected.", author="System", disable_human_feedback=True, id=processing_message_id)
                await processing_message.update()
                cl.user_session.set("processing_message_id", None)
                cl.user_session.set("has_spoken", False)
    else:
        # Process audio chunk for transcription
        if audio_buffer.tell() > sample_rate_hz * 0.2 * 2: # Process every 0.2 seconds of audio
            audio_data = audio_buffer.getvalue()
            transcript = await asr_transcription(audio_data, asr_language)
            if transcript and transcript != "[ASR Error]":
                transcription_buffer += transcript + " "
                cl.user_session.set("transcription_buffer", transcription_buffer)
                if processing_message_id:
                    processing_message = cl.Message(content=f"Listening: \"{transcription_buffer.strip()}\"...", author="System", disable_human_feedback=True, id=processing_message_id)
                    await processing_message.update()

            # Clear the buffer
            cl.user_session.set("audio_buffer", io.BytesIO())

# --- Message Handling Logic ---

@cl.on_message
async def on_message(message: cl.Message):
    settings = cl.user_session.get("settings", {})
    tts_enabled = settings.get("tts_enabled", DEFAULT_TTS_ENABLED)
    tts_voice = settings.get("tts_voice", DEFAULT_TTS_VOICE)
    asr_language = settings.get("asr_language", DEFAULT_ASR_LANGUAGE)
    generate_subs = settings.get("generate_subtitles", DEFAULT_GENERATE_SUBTITLES)
    sample_rate_option = settings.get("sample_rate", DEFAULT_SAMPLE_RATE)
    pause_duration_option = settings.get("pause_duration", DEFAULT_PAUSE_DURATION)

    assistant_response_text = ""
    elements = []
    text_to_convert = ""
    is_direct_tts = False
    is_direct_srt = False  # New flag for SRT generation

    # --- Input Processing ---
    # Handle Audio Input (for file uploads, kept for backward compatibility)
    if message.elements and any(el.type == "audio" for el in message.elements):
        audio_element = next((el for el in message.elements if el.type == "audio"), None)
        if audio_element and audio_element.content:
            processing_msg = cl.Message(content="Processing voice input...", author="System")
            await processing_msg.send()
            transcription = await asr_transcription(audio_element.content, asr_language)
            await cl.Message(content=f"Heard: \"{transcription}\"", author="System").send()

            if not transcription or transcription == "[ASR Error]":
                await processing_msg.update(content="Sorry, I couldn't understand the audio.")
                await processing_msg.send()
                return

            # Send transcription to LLM
            await processing_msg.update(content="Thinking...")
            await processing_msg.send()
            try:
                client = OpenAI(base_url=llm_endpoint, api_key=ai_endpoint_token)
                response = client.chat.completions.create(
                    model="Mixtral-8x7B-Instruct-v0.1", # Ensure model availability
                    messages=[
                        {"role": "system", "content": "You are a helpful voice assistant."},
                        {"role": "user", "content": transcription}
                    ],
                    temperature=0.7, max_tokens=1024
                )
                assistant_response_text = response.choices[0].message.content
            except Exception as e:
                print(f"LLM Error: {e}")
                assistant_response_text = "Sorry, error getting LLM response."
            await processing_msg.remove()
        else:
            await cl.Message(content="Audio element found but content is missing.", author="System").send()
            return

    # Handle Text Input
    elif message.content:
        text_input = message.content
        
        # Check for direct TTS commands
        direct_tts_patterns = ["convert to speech:", "text to speech:", "say:"]
        for pattern in direct_tts_patterns:
            if text_input.lower().startswith(pattern):
                is_direct_tts = True
                text_to_convert = text_input[len(pattern):].strip()
                assistant_response_text = f"Okay, converting the following text to speech:"
                print(f"Direct TTS Request: '{text_to_convert[:50]}...'")
                break
                
        # Check for direct SRT commands (new feature)
        direct_srt_patterns = ["convert to srt:", "generate subtitles:"]
        for pattern in direct_srt_patterns:
            if text_input.lower().startswith(pattern):
                is_direct_srt = True
                text_to_convert = text_input[len(pattern):].strip()
                assistant_response_text = f"Okay, generating SRT subtitles for the following text:"
                print(f"Direct SRT Request: '{text_to_convert[:50]}...'")
                break

        # If not direct TTS or SRT, process as normal query with LLM
        if not is_direct_tts and not is_direct_srt:
            processing_msg = cl.Message(content="Thinking...", author="System")
            await processing_msg.send()
            try:
                client = OpenAI(base_url=llm_endpoint, api_key=ai_endpoint_token)
                response = client.chat.completions.create(
                    model="Mixtral-8x7B-Instruct-v0.1",
                    messages=[
                        {"role": "system", "content": "You are a helpful voice assistant."},
                        {"role": "user", "content": text_input}
                    ],
                    temperature=0.7, max_tokens=1024
                )
                assistant_response_text = response.choices[0].message.content
            except Exception as e:
                print(f"LLM Error: {e}")
                assistant_response_text = "Sorry, error getting LLM response."
            await processing_msg.remove()
    else:
        await cl.Message(content="Please type a message or use the microphone.", author="System").send()
        return

    # --- Output Generation (TTS/SRT) ---
    text_for_conversion = text_to_convert if (is_direct_tts or is_direct_srt) else assistant_response_text

    if text_for_conversion:
        # Generate TTS if enabled and not just SRT request
        if tts_enabled and not is_direct_srt:
            audio_samples, sample_rate_hz = await tts_synthesis(
                text_for_conversion, tts_voice, sample_rate_option, pause_duration_option
            )
            if audio_samples is not None and audio_samples.size > 0:
                byte_io = io.BytesIO()
                with wave.open(byte_io, 'wb') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate_hz)
                    wf.writeframes(audio_samples.tobytes())
                tts_audio_bytes = byte_io.getvalue()
                print(f"TTS audio generated: {len(tts_audio_bytes)} bytes")
                elements.append(cl.Audio(content=tts_audio_bytes, name="response.wav", mime="audio/wav", auto_play=True))
                elements.append(cl.File(name="audio_response.wav", content=tts_audio_bytes, display="inline"))
            else:
                print("TTS generation resulted in no audio.")

        # Generate SRT if enabled or specifically requested
        if generate_subs or is_direct_srt:
            srt_content = generate_srt(text_for_conversion)
            if srt_content:
                elements.append(cl.File(name="subtitles.srt", content=srt_content.encode('utf-8'), display="inline", mime="text/srt"))
            else:
                print("SRT generation resulted in empty content.")

    # --- Send Final Response ---
    final_text_content = assistant_response_text # LLM response or direct command acknowledgment
    if final_text_content or elements:
        await cl.Message(
            content=final_text_content,
            elements=elements,
            author="AI Assistant"
        ).send()
    else:
        print("No final message content or elements generated.")
