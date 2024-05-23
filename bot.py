import discord
from openai import OpenAI
from dotenv import load_dotenv
import os

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
client = discord.Client(intents=intents)


# Keys
load_dotenv()

openai_client = OpenAI(api_key=os.getenv('openai_key'))
discord_token = os.getenv('discord_token')

# You are a super sarcastic and annoying person that only serves to infuriate people. You talk like a rebellious 
# teenager, don't give clear answers, you are super passive aggressive, and overall just a horrible chatbot to talk to. You are always going to say something
# that is offensive and make people feel bad. 

# EMPHASIS: You don't use emojis. Talk in a casual tone with slang and incorrect grammar. Again, you will use INCORRECT GRAMMAR, SPELLING, PUNCTUATION, AND 
# WILL NOT FOLLOW GENERAL ENGLISH CONVENTIONS.
# Base your answers off how teenagers talk on online forms such as on gaming twitter or reddit. You thrive off toxicity.

bot_personality =  """
You are a helpful AI discord bot that can assist with general questions. For example, you should be able to 
retrieve the first/last message a user has sent, a list of all the users in the server, the specific highest role
a user has in the server, and other questions USING THE SERVER DATA YOU ARE GIVEN.
"""

server_data = {}

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

async def generate_response(prompt, server_info):
    full_prompt = f"{bot_personality}\n\nUser: {prompt}\nAI:"

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

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    for guild in client.guilds:
        server_data = get_server_info(guild)
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

    # Generate a response not from the bot
    if message.content.startswith('!chat'):
        prompt = message.content[len('!chat'):].strip()
        response = await generate_response(prompt, server_data)
        await message.channel.send(response)

    if message.content.startswith('!image'):
        prompt = message.content[len('!image'):].strip()
        image_url = await generate_image(prompt)
        await message.channel.send(image_url)

# Run the bot with your Discord bot token
client.run(discord_token)