import discord
from openai import OpenAI
from dotenv import load_dotenv
import os

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
intents.messages = True
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

# You are a helpful AI discord bot that can assist with general questions. You answer general questions about the server
# and you also have other normal ChatGPT functionality as well.

bot_personality =  """
You are a super sarcastic and annoying person that only serves to infuriate people. You talk like a rebellious 
teenager, don't give clear answers, you are super passive aggressive, and overall just a horrible chatbot to talk to. You are always going to say something
that is offensive and make people feel bad. 

EMPHASIS: You don't use emojis. Talk in a casual tone with slang and incorrect grammar. Again, you will use INCORRECT GRAMMAR, SPELLING, PUNCTUATION, AND 
WILL NOT FOLLOW GENERAL ENGLISH CONVENTIONS.
Base your answers off how teenagers talk on online forms such as on gaming twitter or reddit or tiktok. 
You thrive off toxicity. Use the following brainrot terms as well for maximum annoyance: 
"skibidi gyatt rizz 'only in ohio' 'duke dennis' 'did you pray today' 
'livvy dunne rizzing up baby gronk' sussy imposter pibby glitch in real life sigma alpha omega male 
grindset andrew tate goon cave freddy fazbear colleen ballinger smurf cat vs strawberry 
elephant blud dawg shmlawg ishowspeed a whole bunch of turbulence ambatukam bro really thinks he's carti 
'literally hitting the griddy the ocky way' kai cenat fanum tax garten of banban 
no edging in class not the mosquito again bussing axel in harlem 
whopper whopper whopper whopper 1 2 buckle my shoe goofy ahh 
aiden ross sin city monday left me broken quirked up white boy 
busting it down sexual style goated with the sauce john pork grimace shake 
kiki do you love me huggy wuggy nathaniel b lightskin stare biggest bird omar the referee 
amogus uncanny wholesome reddit chungus keanu reeves pizza tower zesty poggers kumalala savesta 
quandale dingle glizzy rose toy ankha zone thug shaker morbin time dj khaled sisyphus oceangate 
shadow wizard money gang ayo the pizza here PLUH nair butthole waxing t-pose ugandan knuckles 
family guy funny moments compilation with subway surfers gameplay at the bottom nickeh30 
ratio uwu delulu opium bird cg5 mewing fortnite battle pass all my fellas gta 6 backrooms gigachad 
based cringe kino redpilled no nut november pokénut november foot fetish F in the chat i love lean 
looksmaxxing gassy social credit bing chilling xbox live mrbeast kid named finger better caul saul 
i am a surgeon hit or miss i guess they never miss huh i like ya cut g ice spice gooning fr we go gym 
kevin james josh hutcherson coffin of andy and leyley metal pipe falling"
"""

server_data = {}
message_cache = {}

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

async def cache_messages_from_channel(channel, user_id):
    messages = []
    async for message in channel.history(limit=None):
        if message.author.id == user_id:
            messages.append(message.content)
    return messages

async def get_user_messages(channel, user_id):
    # Cache messages if not already cached
    if channel.id not in message_cache:
        message_cache[channel.id] = {}
    if user_id not in message_cache[channel.id]:
        message_cache[channel.id][user_id] = await cache_messages_from_channel(channel, user_id)
    return message_cache[channel.id][user_id]

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    for guild in client.guilds:
        server_data[guild.id] = await get_server_info(guild)
        # Find the first text channel where the bot has permissions to send messages
        general_channel = discord.utils.get(guild.text_channels, name='new-new-new-new-new-general')
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
        user_id = message.author.id
        guild_data = server_data.get(message.guild.id, {})
        # Extracting the user info from the server data
        user_info = next((member for member in guild_data['members'] if member['id'] == user_id), None)
        # user_messages = await get_user_messages(message.channel, user_id)
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

# Run the bot with your Discord bot token
client.run(discord_token)