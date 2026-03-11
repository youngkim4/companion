import asyncio

import discord
from discord.ext import commands

from config import DISCORD_TOKEN, log

COGS = [
    "cogs.chat",
    "cogs.image",
    "cogs.voice",
    "cogs.members",
]

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
intents.messages = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents)
bot.server_data = {}


async def load_cogs():
    for cog in COGS:
        try:
            await bot.load_extension(cog)
            log.info("Loaded cog: %s", cog)
        except Exception as e:
            log.error("Failed to load cog %s: %s", cog, e)


async def main():
    async with bot:
        await load_cogs()
        await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
