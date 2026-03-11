import base64
from io import BytesIO

import discord
from discord.ext import commands
from openai import AsyncOpenAI

from config import MAX_PROMPT_LENGTH, OPENAI_KEY, log

openai_client = AsyncOpenAI(api_key=OPENAI_KEY)


async def generate_image(prompt):
    try:
        response = await openai_client.images.generate(
            model="gpt-image-1.5",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        image_bytes = base64.b64decode(response.data[0].b64_json)
        return discord.File(BytesIO(image_bytes), filename="image.png")
    except Exception as e:
        log.error("Failed to generate image: %s", e)
        return None


class Image(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def image(self, ctx, *, prompt: str):
        if len(prompt) > MAX_PROMPT_LENGTH:
            await ctx.send(
                f"Prompt too long. Please keep it under {MAX_PROMPT_LENGTH} characters."
            )
            return

        result = await generate_image(prompt)
        if result:
            await ctx.send(file=result)
        else:
            await ctx.send("Sorry, I couldn't generate an image.")

    @image.error
    async def image_error(self, ctx, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send("Please provide a prompt after !image")
        else:
            log.error("Image command error: %s", error)


async def setup(bot):
    await bot.add_cog(Image(bot))
