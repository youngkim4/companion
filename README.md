## Companion AI Discord Bot

## This project is a Discord bot that functions as an AI-powered voice assistant. It can transcribe speech from users in a voice channel, respond using GPT-based text-to-speech (TTS), and interact with users through text commands.

## Features
1. **Voice Transcription**: 
   - The bot joins a voice channel and listens to users speaking.
   - Uses OpenAI's Whisper model for transcription.

2. **Text-to-Speech Responses**:
   - Generates intelligent and contextually appropriate responses using GPT models.
   - Plays audio responses back in the voice channel using OpenAI TTS.

3. **Text Commands**:
   - `!join`: Join the user's voice channel and start listening.
   - `!leave`: Leave the voice channel.
   - `!chat <message>`: Respond to user messages in the text channel.
   - `!image <prompt>`: Generate and return an image based on the given prompt.

4. **AI Integration**:
   - Leverages OpenAI Whisper for transcription.
   - Uses OpenAI GPT-4 for text generation.
   - Includes OpenAI TTS for audio playback.
