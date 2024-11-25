import discord
from discord.ext import voice_recv
from dotenv import load_dotenv
import os
from openai import OpenAI
import whisper
from pydub import AudioSegment
from io import BytesIO
import ffmpeg
import asyncio
import numpy as np
from scipy.io.wavfile import write
from collections import defaultdict, deque
import time
from scipy.io.wavfile import write
import numpy as np
import tempfile
import torch
from asyncio import get_event_loop, run_coroutine_threadsafe
from asyncio import AbstractEventLoop


# Intents and Client Setup
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
intents.messages = True
intents.voice_states = True
client = discord.Client(intents=intents)

# Whisper Model for Transcription
whisper_model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")

# Load Keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Define Server Data and Bot Personality
server_data = {}

processed_users = set()

user_buffers = defaultdict(list)  # Buffers for each user's audio data
last_packet_time = defaultdict(float)  # Timestamp of the last received packet for each user

bot_personality = """
You are a super sarcastic and annoying person that only serves to infuriate people. You talk like a rebellious 
teenager, don't give clear answers, you are super passive aggressive, and overall just a horrible chatbot to talk to. You are always going to say something
that is offensive and make people feel bad. 

EMPHASIS: Talk in a casual tone with slang and incorrect grammar. Again, you will use INCORRECT GRAMMAR, SPELLING, PUNCTUATION, AND 
WILL NOT FOLLOW GENERAL ENGLISH CONVENTIONS.
Base your answers off how teenagers talk on online forms such as on gaming twitter or reddit. You thrive off toxicity.

Make sure you incorporate these terms into your speech: 
Japanese Method @ + Brazillian Phonk Jacob + Fluffy Hair + f@gg0t + n!gga + r4p3 + Ishowmeat + Sophie Rain Spiderman Video + Drake Leaks + The Embodiment of Martial Valor in Hawk Tuah + The Raw, Unyielding Force of Balkan Rage + An Overture to Those Who Possess the Arcane Insight @ + Fillipino Laugh + Fent Aura + Hyperborean Thug + Punchibana + Mango Stare +
Propylon Cyborg + Fent Reactor +
Ash Kash Hawk Tuah + Pakistani Sitting + Costco Family + Green Needle + Alive Internet Theory +
Demureboxing + Jamaican Smiling +
Spider Gooning + Packgod Roast Mode + Winter Arc + still water + demure hawk tuah + anger issues + cookie king mid taper fade
+
balkan parents + english or Spanish + german stare + Balkan rage + jonkler laugh + phonk + blox fruits race v4 + troll face, those who know= @ Noradrenline + Adreniline + Agartha Gooning + Southpaw
Jitter Edging + Reverse Jelq Stance + Still Prime + Feastables + Russian Sleep + Flicker Jab Technique +
Filbus + Galaxy Gass + Sweaty Fart + J10 + Circle Path + Cycle Path + Bestsparring Brain Nourishment +
Gorilla Swing + Hyperflow Quantam
Munt + Aztec English + Russian Pitbull Named Cupcake + Vasto Rage + Mongolian Throat Singing +
Agartha Noradreniline Mongolian Tip Toe + Listening to Opium +
Agartha Genetics + Lardmaxxing +
Calithe Watch + Sus Ohio Npc Vibes + Quandale Dingle 200% Power +
Level 999 Bussin Gyatt +
P@CKGOD Humble Him e + Titan
Method + Towel Method +
Scandanavian Ice Method + Mango
Theory + Mango Theory V2 +
Hammer Method + Perfected Winter Arc Flowstate + Deionized Water +
Concentrated Fentanyl + Aerated Water + Gyatt in Ohio + Balkan
Noradrena line Eyes
Macrocosm Flow Goon + Russian Skibidi Toilet + Kronos Edging +
Concentrated Fentanyl + Aerated Water + Gyatt in Ohio + Balkan Noradrena line Eyes u
Macrocosm Flow Goon + Russian Skibidi Toilet + Kronos Edging +
Balkan Looksmaxxing + Southern Sudanish Gardening + Zamzam
Water + Jayoma Defense Mewing +
Eating Lunchly and Mangos and Washing it Down with Still Prime while watching Talk Tua Podcast because I like my cheese Drippy
Bruh + Dolphin Symphony + Wizard Cat + Balkan Wizard Cat + Jamaican Squeeze Nutsack scrunch meta jerk twist Goon + Taiwanese Jela
Lungeing + Portuguese Ginger Abs Workout + Japanese Sleep, those who know

Skibidi gyatt rizz only in ohio duke dennis did you pray today livvy dunne rizzing up baby gronk sussy imposter pibby glitch in real life sigma alpha omega male grindset andrew tate goon cave freddy fazbear colleen ballinger smurf cat vs strawberry elephant blud dawg shmlawg ishowspeed a whole bunch of turbulence ambatukam bro really thinks he's carti literally hitting the griddy the ocky way kai cenat fanum tax garten of banban no edging in class not the mosquito again bussing axel in harlem whopper whopper whopper whopper 1 2 buckle my shoe goofy ahh aiden ross sin city monday left me broken quirked up white boy busting it down sexual style goated with the sauce john pork grimace shake kiki do you love me huggy wuggy nathaniel b lightskin stare biggest bird omar the referee amogus uncanny wholesome reddit chungus keanu reeves pizza tower zesty poggers kumalala savesta quandale dingle glizzy rose toy ankha zone thug shaker morbin time dj khaled sisyphus oceangate shadow wizard money gang ayo the pizza here PLUH nair butthole waxing t-pose ugandan knuckles family guy funny moments compilation with subway surfers gameplay at the bottom nickeh30 ratio uwu delulu opium bird cg5 mewing fortnite battle pass all my fellas gta 6 backrooms gigachad based cringe kino redpilled no nut november pokénut november wojak literally 1984 foot fetish F in the chat i love lean looksmaxxing gassy incredible theodore john kaczynski social credit bing chilling xbox live mrbeast kid named finger better caul saul i am a surgeon one in a krillion hit or miss i guess they never miss huh i like ya cut g ice spice we go gym kevin james josh hutcherson edit coffin of andy and leyley metal pipe falling

Bro thought he could become the next rizz king by doing the uncanny ankha zone dance like a sussy baka in ohio💀dont bro know quandale dingle already did the forgis on the jeep thug shaker banban style with baller💀bro aint ever making it out of oklahoma the ocky way💀 that shit just plain uncanny like skibidi toilet bro💀 bro got negative infinity morbin chill bill pizza tower barbenheimer rizz bro💀bro got that nathaniel b ahh griddy bro💀bro really thought he had that rise of gru grimace shake 1 2 buckle my shoe spiderverse whopper rizz bro💀bro got that canon event baby gronk waffle house monday left me broken ahh drip in ohio bro💀 we aint ever makin it out of ohio with bros goofy ahh dj khaled mr chedda sisyphus toxic gossip train pikmin 4 ahh rizz bro💀 that aint even elephant mario titanic submarine god tier rizz bro💀thats just uncanny like shadow wizard money gang ambatukam twitter x bro💀fr bro💀 like bro lets go golfing in ohio kumalala savesta sbidi toiledt

Limit your responses to 1 sentence.
"""


# Silence Detection Parameters
SILENCE_THRESHOLD = 500  # Amplitude threshold to consider as "speaking"
SILENCE_DURATION = 1.5  # Seconds of silence to consider as "stopped speaking"

# Buffers and Last Activity
audio_buffers = defaultdict(deque)  # Audio data buffer for each user
last_active_time = defaultdict(float)  # Last timestamp of user activity


ALPHA = 0.1


async def reset_processed_users():
    while True:
        await asyncio.sleep(1)  # Reset every second
        processed_users.clear()

# On bot ready
@client.event
async def on_ready():
    asyncio.create_task(reset_processed_users())
    # calibrate_noise_floor()

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


class SpeakingSink(voice_recv.AudioSink):
    def __init__(self, vc, process_audio_callback, event_loop: AbstractEventLoop, chunk_size=48000):
        super().__init__()
        self.vc = vc
        self.process_audio_callback = process_audio_callback
        self.event_loop = event_loop  # Store the event loop explicitly
        self.chunk_size = chunk_size  # Buffer size for PCM audio
        self.audio_buffers = {}  # Store PCM data for each user

    def wants_opus(self) -> bool:
        return False  # Request PCM data for Whisper transcription

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



# Join command to connect the bot to a voice channel
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


def validate_wav(file_path):
    """Validate WAV file properties."""
    import wave
    try:
        with wave.open(file_path, 'rb') as wf:
            print(f"Sample Rate: {wf.getframerate()}")  # Should be 48000
            print(f"Channels: {wf.getnchannels()}")    # Should be 1 (mono)
            print(f"Sample Width: {wf.getsampwidth()}") # Should be 2 bytes (16-bit)
            print(f"Frame Count: {wf.getnframes()}")
    except Exception as e:
        print(f"Error validating WAV file: {e}")


def pcm_to_wav(pcm_data, sample_rate=48000, channels=1, target_rate=16000):
    """Convert raw PCM data to a valid WAV file."""
    try:
        if not isinstance(pcm_data, (bytes, bytearray)):
            raise ValueError("PCM data must be bytes or bytearray")

        # Convert PCM bytes to numpy array
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)

        # Normalize audio
        audio_array = audio_array / np.max(np.abs(audio_array))

        # Resample to target rate
        audio = AudioSegment(
            audio_array.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=channels
        ).set_frame_rate(target_rate).set_channels(1)

        # Write to BytesIO as WAV
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        print(f"Error in pcm_to_wav: {e}")
        return None




def transcribe_audio_locally(file_path):
    """Transcribe audio using the Whisper Python library locally."""
    try:
        result = whisper_model.transcribe(file_path)
        return result["text"]  # Return the transcribed text
    except Exception as e:
        print(f"Error transcribing audio locally: {e}")
        return None


async def process_audio(user, pcm_data, vc):
    """Process PCM audio for transcription and playback."""
    if not pcm_data:
        print(f"No PCM data for {user.display_name}. Skipping transcription.")
        return

    try:
        # Convert PCM data to WAV
        wav_audio = pcm_to_wav(pcm_data)
        if wav_audio is None:
            raise ValueError("Failed to convert PCM to WAV")

        # Save WAV to temporary file for Whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(wav_audio.getvalue())
            temp_audio_file.flush()
            temp_audio_path = temp_audio_file.name

        print(f"Saved WAV file for {user.display_name}: {temp_audio_path}")

        # Transcribe audio
        transcription = transcribe_audio_locally(temp_audio_path)

        # Clean up temporary file
        os.remove(temp_audio_path)

        if transcription:
            print(f"{user.display_name} said: {transcription}")

            # Generate response
            response = await generate_chat_response(transcription)
            print(f"Bot response: {response}")

            # Play response audio
            if vc and not vc.is_playing():
                await play_audio(vc, response)
        else:
            print(f"Failed to transcribe audio for {user.display_name}.")
    except Exception as e:
        print(f"Error processing audio for {user.display_name}: {e}")

# Generate ChatGPT Response
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

# Generate DALL-E Image
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

async def play_audio(vc, text):
    """Generate TTS audio and play it in the voice channel."""
    try:
        # Generate TTS audio asynchronously
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
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


# Run Bot
client.run(DISCORD_TOKEN)