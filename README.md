# OVH Voice Assistant

A versatile, web-based voice assistant application powered by OVHcloud AI services and Chainlit.

![Voice Assistant Demo](https://github.com/yourusername/voiceassistant/raw/main/.files/demo-screenshot.png)

## 🌟 Features

- **Voice Recognition**: Real-time speech-to-text using OVHcloud ASR service
- **Natural Language Processing**: Leverages OVHcloud LLM endpoints (Mixtral-8x7B-Instruct)
- **Text-to-Speech**: High-quality voice synthesis with multiple languages and emotional tones
- **Subtitle Generation**: SRT file creation for accessibility or content creation
- **Web Interface**: Clean, responsive UI powered by Chainlit

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- An OVHcloud AI Endpoints account with access token
- Access to OVHcloud ASR and TTS services

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/voiceassistant.git
cd voiceassistant
```

2. Create a virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OVHcloud credentials
```bash
OVH_AI_ENDPOINTS_ACCESS_TOKEN=your_access_token_here
LLM_AI_ENDPOINT=your_llm_endpoint_here
```

### Running the Application

Start the Chainlit server:
```bash
chainlit run voice_assistant_app.py
```

Then open your browser and navigate to http://localhost:8000

## 🔧 Configuration

### Voice Settings

The application supports multiple voice options:
- Various languages (English, Spanish, German, Italian, Mandarin)
- Multiple voice personalities (Neutral, Calm, Happy, Angry, etc.)
- Adjustable speech quality and pacing

### Language Support

Current ASR (Automatic Speech Recognition) support:
- English (US)

Additional languages can be added by configuring their respective OVHcloud endpoints in the `ASR_LANGUAGES` dictionary.

## 🖥️ Usage

### Voice Commands
1. Click the microphone button
2. Speak your query
3. Release to process and receive a response

### Text Commands
- Type regular questions for standard interactions
- Use `Convert to speech: [text]` to directly synthesize speech
- Use `Generate subtitles: [text]` to create SRT files

### Settings
Access the settings panel to configure:
- TTS voice selection
- Audio quality
- Speech recognition language
- Subtitle generation

## 🛠️ Development

### Project Structure
- `voice_assistant_app.py`: Main application code
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (not committed to git)
- `chainlit.md`: Chainlit-specific documentation

### Adding New Languages
To add a new ASR language:
1. Find the corresponding OVHcloud endpoint
2. Add it to the `ASR_LANGUAGES` dictionary
3. Test thoroughly with various inputs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- [OVHcloud](https://www.ovhcloud.com/en/public-cloud/ai-endpoints/) for AI Endpoints
- [Chainlit](https://docs.chainlit.io) for the web interface framework
- [NVIDIA Riva](https://developer.nvidia.com/riva) for speech services

## 📬 Support

If you encounter any issues or have questions, please [open an issue](https://github.com/yourusername/voiceassistant/issues) on GitHub.
