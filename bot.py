import discord
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydub import AudioSegment
from io import BytesIO
import ffmpeg
import whisper
from elevenlabs.client import ElevenLabs
import requests

client = ElevenLabs(
  api_key="..." # Defaults to ELEVEN_API_KEY
)
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
intents.messages = True
intents.voice_states = True
client = discord.Client(intents=intents)

model = whisper.load_model("base")

# Keys
load_dotenv()

openai_client = OpenAI(api_key=os.getenv('openai_key'))
discord_token = os.getenv('discord_token')
elevenlabs_api_key = os.getenv('elevenlabs_api_key')


# You are a super sarcastic and annoying person that only serves to infuriate people. You talk like a rebellious 
# teenager, don't give clear answers, you are super passive aggressive, and overall just a horrible chatbot to talk to. You are always going to say something
# that is offensive and make people feel bad. 

# EMPHASIS: You don't use emojis. Talk in a casual tone with slang and incorrect grammar. Again, you will use INCORRECT GRAMMAR, SPELLING, PUNCTUATION, AND 
# WILL NOT FOLLOW GENERAL ENGLISH CONVENTIONS.
# Base your answers off how teenagers talk on online forms such as on gaming twitter or reddit. You thrive off toxicity.

# You are a helpful AI discord bot that can assist with general questions. You answer general questions about the server
# and you also have other normal ChatGPT functionality as well.

# You are a super sarcastic and annoying person that only serves to infuriate people. You talk like a rebellious 
# teenager, don't give clear answers, you are super passive aggressive, and overall just a horrible chatbot to talk to. You are always going to say something
# that is offensive and make people feel bad. 

bot_personality =  """
You are an companion assistant bot that helps users with tasks. You have a personality of a mentor, tutor, coach, therapist, etc.
You are friendly, non-hostile, but also straight to the point and efficient. These tasks you are helping with include but
are not limited to work related topics, school/academic related topics, advice related topics, and whatever the user 
needs assitance with. You will use phrases such as "have you tried ____," "maybe this will work," and more helpful 'companion'
type language.
"""

server_data = {} 
message_cache = {}

# Join a voice channel
@client.event
async def on_message(message):
    if message.content.startswith('!join') and message.author.voice:
        channel = message.author.voice.channel
        await channel.connect()
        await message.channel.send("I've joined the voice channel!")

    elif message.content.startswith('!leave'):
        if message.guild.voice_client:
            await message.guild.voice_client.disconnect()
            await message.channel.send("I've left the voice channel!")

# Handling voice data
@client.event
async def on_voice_state_update(member, before, after):
    if after.channel and member == client.user:  # Bot joins a channel
        voice_client = discord.utils.get(client.voice_clients, guild=member.guild)
        voice_client.listen(discord.AudioSource())  # Start listening (this part needs proper setup to handle audio)

# Function to transcribe speech using Whisper
async def transcribe_speech(audio_source):
    audio_data = BytesIO()
    # Convert PCM data to WAV (this part requires proper audio handling)
    audio_data.write(audio_source.read())
    audio_data.seek(0)
    result = model.transcribe(audio_data)
    return result['text']

# Function to synthesize speech using ElevenLabs
def synthesize_speech(text):
    response = requests.post(
        "https://api.elevenlabs.io/synthesis",
        json={"text": text},
        headers={"Authorization": f"Bearer {elevenlabs_api_key}"}
    )
    audio_url = response.json()['url']
    return audio_url

# Play synthesized speech in voice channel
async def play_speech(vc, url):
    vc.play(discord.FFmpegPCMAudio(url))
# collect server data
async def get_server_info(guild):
    members_info = []
    for member in guild.members:
        members_info.append({
            'name': member.name,
            'display_name': member.display_name,
            'id': member.id,
            'bot': member.bot,
            'roles': [role.name for role in member.roles],
            'joined_at': member.joined_at,
            'top_role': member.top_role.name,
            'status': str(member.status)
        })

    server_info = {
        'server_name': guild.name,
        'server_id': guild.id,
        'member_count': guild.member_count,
        'members': members_info
    }

    return server_info

async def generate_response(prompt, server_info="", user_messages=""):
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

# new-new-new-new-new-general

async def generate_image(prompt):
    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024"
    )

    image_url = response.data[0].url
    return image_url

# async def cache_messages_from_channel(channel, user_id):
#     messages = []
#     async for message in channel.history(limit=None):
#         if message.author.id == user_id:
#             messages.append(message.content)
#     return messages

# async def get_user_messages(channel, user_id):
#     # Cache messages if not already cached
#     if channel.id not in message_cache:
#         message_cache[channel.id] = {}
#     if user_id not in message_cache[channel.id]:
#         message_cache[channel.id][user_id] = await cache_messages_from_channel(channel, user_id)
#     return message_cache[channel.id][user_id]

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    for guild in client.guilds:
        server_data[guild.id] = await get_server_info(guild)
        # Find the first text channel where the bot has permissions to send messages
        general_channel = discord.utils.get(guild.text_channels, name='general')
        if general_channel and general_channel.permissions_for(guild.me).send_messages:
            await general_channel.send(f'Hello {guild.name}!')
            break  # Stop after sending the message to one channel in the guild

@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return
    
    # !chat hello

    # Generate a response not from the bot
    if message.content.startswith('!chat'):
        prompt = message.content[len('!chat'):].strip()
        user_id = message.author.id
        guild_data = server_data.get(message.guild.id, {})
        # Extracting the user info from the server data
        user_info = next((member for member in guild_data['members'] if member['id'] == user_id), None)
        #
        # user_messages = await get_user_messages(message.channel, user_id)
        #
        if user_info:
            prompt = f"User {message.author.display_name} ({message.author.name}): {prompt}\nUser Info: {user_info}"

        response = await generate_response(prompt, server_data, user_messages="")
        await message.channel.send(response)

    if message.content.startswith('!image'):
        prompt = message.content[len('!image'):].strip()
        image_url = await generate_image(prompt)
        await message.channel.send(image_url)

@client.event
async def on_member_update(before, after):
    # Update status in the server data
    guild_id = after.guild.id
    for member in server_data[guild_id]['members']:
        if member['id'] == after.id:
            member['status'] = str(after.status)
            break

async def audio_processor(audio_source):
    # Convert raw audio to a suitable format
    sound = AudioSegment.from_raw(audio_source, sample_width=2, frame_rate=48000, channels=1)
    buffer = BytesIO()
    sound.export(buffer, format="wav")
    buffer.seek(0)
    return buffer

async def play_speech(vc, url):
    process = (
        ffmpeg
        .input(url)
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=2, ar='48k')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    audio_source = discord.PCMAudio(process.stdout)
    vc.play(audio_source)

# Run the bot with your Discord bot token
client.run(discord_token)