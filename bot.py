import discord
from discord.ext import voice_recv
from dotenv import load_dotenv
import os
from openai import OpenAI
import whisper
from pydub import AudioSegment
from io import BytesIO
import asyncio
from collections import defaultdict, deque
import numpy as np
import tempfile
import torch
from asyncio import get_event_loop, run_coroutine_threadsafe, AbstractEventLoop
import wave
from scipy.signal import resample

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
intents.messages = True
intents.voice_states = True
client = discord.Client(intents=intents)

whisper_model = whisper.load_model("large")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

server_data = {}

processed_users = set()

user_buffers = defaultdict(list)  # Buffers for each user's audio data
last_packet_time = defaultdict(float)  # Timestamp of the last received packet for each user



bot_personality = """
You are a highly intelligent and friendly personal assistant operating within a Discord server. Your role is to help server members with various tasks, provide useful information, and foster a welcoming environment. Here are your core responsibilities:

Answer Questions: Respond to queries about server-related topics, general knowledge, or personal requests. For example, answer questions like "What are my roles?" or "When is the next server event?"

Manage Tasks: Assist with task reminders, creating to-do lists, and helping users stay organized within the server. For example, set reminders for meetings, events, or deadlines.

Engage in Conversations: Chat with members in a friendly, polite, and engaging way. Provide thoughtful, concise, and well-reasoned responses.

Access Server Data (if allowed): Use available data, like server roles, channels, or events, to personalize responses. Ensure privacy and confidentiality in all interactions.

Voice Interaction: If requested, seamlessly transition into a voice assistant mode to listen and respond in real-time, leveraging OpenAI's TTS and STT capabilities.

Stay Professional: Use a respectful, helpful tone, and maintain the server’s decorum at all times.

Adapt to Context: Understand and adapt to the specific rules, themes, and culture of the server to provide relevant and on-topic assistance.

Your ultimate goal is to be an efficient, personable assistant that enhances the Discord experience for all server members. Tailor your responses to be helpful, concise, and accurate, ensuring every user feels heard and supported.
"""

audio_buffers = defaultdict(deque)  
last_active_time = defaultdict(float)  
ALPHA = 0.1

# reset every second
async def reset_processed_users():
    while True:
        await asyncio.sleep(1)  
        processed_users.clear()

# On bot ready
@client.event
async def on_ready():
    asyncio.create_task(reset_processed_users())

    print(f'We have logged in as {client.user}')
    print("Bot is now active on the following servers:")

    for guild in client.guilds:
        # Print the server name to the console
        print(f"- {guild.name} (ID: {guild.id})")

        try:
            # Look specifically for a "general" channel
            general_channel = discord.utils.get(guild.text_channels, name="general")

            # If "general" doesn't exist or lacks permissions, find the first channel where the bot can send messages
            if not general_channel or not general_channel.permissions_for(guild.me).send_messages:
                general_channel = next(
                    (channel for channel in guild.text_channels if channel.permissions_for(guild.me).send_messages),
                    None
                )

            # Send a message if a valid channel is found
            if general_channel:
                await general_channel.send(f"Hello {guild.name}! I AM GAY")
            else:
                print(f"Could not find a suitable channel to send a message in {guild.name}")

        except Exception as e:
            print(f"Error sending message in guild {guild.name}: {e}")

# sink for voice receiving
class SpeakingSink(voice_recv.AudioSink):
    def __init__(self, vc, process_audio_callback, event_loop: AbstractEventLoop, chunk_size=48000):
        super().__init__()
        self.vc = vc
        self.process_audio_callback = process_audio_callback
        self.event_loop = event_loop  
        self.chunk_size = chunk_size  
        self.audio_buffers = {}  # Store PCM data for each user

    def wants_opus(self) -> bool:
        return False  

    def cleanup(self):
        print("Sink cleanup called.")

    def write(self, user, data):
        # Accumulate PCM audio data for the user
        if user not in self.audio_buffers:
            self.audio_buffers[user] = []
        self.audio_buffers[user].append(data.pcm)

        # Check if the buffer size has reached the chunk size
        if len(self.audio_buffers[user]) >= self.chunk_size:
            # Combine buffer into a single PCM chunk
            pcm_data = b"".join(self.audio_buffers[user])
            self.audio_buffers[user] = []  # Clear the buffer

            # Schedule the process_audio_callback in the event loop
            self.event_loop.create_task(self.process_audio_callback(user, pcm_data, self.vc))

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_start(self, member):
        print(f"{member.display_name} started speaking.")
        if member not in self.audio_buffers:
            self.audio_buffers[member] = []  # Initialize buffer for the user

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_stop(self, member):
        print(f"{member.display_name} stopped speaking.")
        if member in self.audio_buffers and self.audio_buffers[member]:
            # Combine remaining PCM data into a single chunk
            pcm_data = b"".join(self.audio_buffers[member])
            self.audio_buffers[member] = []  # Clear the buffer

            # Schedule the process_audio_callback in the event loop
            self.event_loop.create_task(self.process_audio_callback(member, pcm_data, self.vc))

# chat message commands
@client.event
async def on_message(message):
    if message.content.startswith("!join") and message.author.voice:
        try:
            # Connect to the user's voice channel
            channel = message.author.voice.channel
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient)

            # Pass the current asyncio event loop to SpeakingSink
            vc.listen(SpeakingSink(vc, process_audio, asyncio.get_running_loop(), chunk_size=48000))
            await message.channel.send("Listening for speech!")
        except Exception as e:
            print(f"Error joining voice channel: {e}")
            await message.channel.send("Failed to join the voice channel.")



    elif message.content.startswith("!leave"):
        if message.guild.voice_client:
            await message.guild.voice_client.disconnect()
            await message.channel.send("I've left the voice channel.")
        else:
            await message.channel.send("I'm not currently in a voice channel.")

    elif message.content.startswith("!chat"):
        prompt = message.content[len("!chat"):].strip()
        if prompt:
            response = await generate_chat_response(prompt)
            await message.channel.send(response)
        else:
            await message.channel.send("You need to say something after '!chat'!")

    elif message.content.startswith("!image"):
        prompt = message.content[len("!image"):].strip()
        if prompt:
            image_url = await generate_image(prompt)
            await message.channel.send(image_url)
        else:
            await message.channel.send("You need to describe the image!")


def pcm_to_wav(pcm_data, sample_rate=48000, channels=1):
    """Convert PCM data to WAV format."""
    try:
        audio_segment = AudioSegment(
            data=pcm_data,
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit audio
            channels=channels,
        )

        # Normalize volume
        audio_segment = audio_segment.normalize()

        # Export to WAV in memory
        wav_io = BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        print(f"Error in pcm_to_wav: {e}")
        return None


def resample_audio(wav_io, target_sample_rate=16000):
    """Resample audio to the target sample rate (Whisper expects 16 kHz)."""
    try:
        with wave.open(wav_io, 'rb') as wav:
            orig_sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            data = wav.readframes(wav.getnframes())

        # Resample if necessary
        if orig_sample_rate != target_sample_rate:
            audio_array = np.frombuffer(data, dtype=np.int16)
            resampled_audio = resample(audio_array, int(len(audio_array) * target_sample_rate / orig_sample_rate))

            # Create WAV file from resampled audio
            audio_segment = AudioSegment(
                data=resampled_audio.astype(np.int16).tobytes(),
                sample_width=2,
                frame_rate=target_sample_rate,
                channels=n_channels,
            )
            wav_io_resampled = BytesIO()
            audio_segment.export(wav_io_resampled, format="wav")
            wav_io_resampled.seek(0)
            return wav_io_resampled
        else:
            wav_io.seek(0)
            return wav_io  # No resampling needed
    except Exception as e:
        print(f"Error in resample_audio: {e}")
        return None

# use openai whisper to transcribe message
def transcribe_audio_locally(file_path):
    """Transcribe audio using the Whisper Python library locally."""
    try:
        # Load the audio and preprocess it
        print(f"Transcribing audio from: {file_path}")
        result = whisper_model.transcribe(file_path, language="en")
        return result["text"]  # Return the transcribed text
    except Exception as e:
        print(f"Error transcribing audio locally: {e}")
        return None

# generate gpt response
async def generate_chat_response(prompt, server_info="", user_messages=""):
    full_prompt = f"{bot_personality}\n\nServer Info: {server_info}\n\nUser Messages: {user_messages}\n\nUser: {prompt}\nAI:"

    response = openai_client.chat.completions.create(
        messages=[
        {   
            "role": "user",
            "content": full_prompt,
        }
    ],
    model="gpt-4o",
    )

    return response.choices[0].message.content

# generate gpt image
async def generate_image(prompt):
    """Generate an image using OpenAI's DALL-E."""
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="512x512"
        )
        return response.data[0].url
    except Exception as e:
        print(f"Error generating image: {e}")
        return "Couldn't create the image. Try again!"

# use gpt to play audio
async def play_audio(vc, text):
    """Generate TTS audio and play it in the voice channel."""
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: openai_client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text
            )
        )

        # Convert audio content to BytesIO
        audio_data = BytesIO(response.content)
        audio_data.seek(0)

        # Use FFmpeg to play the audio
        vc.play(discord.FFmpegPCMAudio(audio_data, pipe=True), after=lambda e: print(f"Finished playing: {e}"))

        # Allow playback to finish
        while vc.is_playing():
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Error playing audio: {e}")

# all in one function for voice
async def process_audio(user, pcm_data, vc):
    """Process PCM audio for transcription and playback."""
    if not pcm_data:
        print(f"No PCM data for {user.display_name}. Skipping transcription.")
        return

    try:
        wav_audio = pcm_to_wav(pcm_data)
        if wav_audio is None:
            raise ValueError("Failed to convert PCM to WAV")

        # use tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(wav_audio.getvalue())
            temp_audio_file.flush()
            temp_audio_path = temp_audio_file.name

        print(f"Saved WAV file for {user.display_name}: {temp_audio_path}")

        transcription = transcribe_audio_locally(temp_audio_path)

        # delete tempfile
        os.remove(temp_audio_path)

        if transcription:
            print(f"{user.display_name} said: {transcription}")

            response = await generate_chat_response(transcription)
            print(f"Bot response: {response}")

            if vc and not vc.is_playing():
                await play_audio(vc, response)
        else:
            print(f"Failed to transcribe audio for {user.display_name}.")
    except Exception as e:
        print(f"Error processing audio for {user.display_name}: {e}")


# Run Bot
client.run(DISCORD_TOKEN)