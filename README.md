# Companion

A Discord bot that acts as a mentor, tutor, and coach. Chat with it, generate images, or talk to it in voice channels.

## Commands

| Command | Description |
|---------|-------------|
| `!chat <message>` | Get a conversational response |
| `!image <prompt>` | Generate an image |
| `!join` | Bot joins your voice channel |
| `!leave` | Bot leaves the voice channel |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires API keys from:
- [Discord](https://discord.com/developers/applications)
- [OpenAI](https://platform.openai.com)
- [ElevenLabs](https://elevenlabs.io)

Add them to a `.env` file, then run:

```bash
python bot.py
```

## Stack

- **Chat & Image Generation**: OpenAI
- **Text-to-Speech**: ElevenLabs
- **Speech-to-Text**: OpenAI Whisper API
- **Bot Framework**: discord.py
