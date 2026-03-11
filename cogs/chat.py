import discord
from discord.ext import commands
from openai import AsyncOpenAI

from config import BOT_PERSONALITY, MAX_PROMPT_LENGTH, OPENAI_KEY, log

openai_client = AsyncOpenAI(api_key=OPENAI_KEY)


async def generate_response(prompt, user_context=""):
    try:
        response = await openai_client.responses.create(
            model="gpt-5.4",
            instructions=BOT_PERSONALITY,
            input=f"Context: {user_context}\n\nUser: {prompt}",
        )
        return response.output_text
    except Exception as e:
        log.error("Failed to generate response: %s", e)
        return f"Sorry, I couldn't generate a response. Error: {e}"


class Chat(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def chat(self, ctx, *, prompt: str):
        if len(prompt) > MAX_PROMPT_LENGTH:
            await ctx.send(
                f"Message too long. Please keep it under {MAX_PROMPT_LENGTH} characters."
            )
            return

        guild_data = self.bot.server_data.get(ctx.guild.id, {})
        user_context = _get_user_context(guild_data, ctx.author.id)
        prompt_with_user = (
            f"User {ctx.author.display_name} ({ctx.author.name}): {prompt}"
        )

        response = await generate_response(prompt_with_user, user_context)
        await ctx.send(response)

    @chat.error
    async def chat_error(self, ctx, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send("Please provide a message after !chat")
        else:
            log.error("Chat command error: %s", error)


def _get_user_context(guild_data, user_id):
    if not guild_data or 'members' not in guild_data:
        return ""
    user_info = next(
        (m for m in guild_data['members'] if m['id'] == user_id),
        None,
    )
    if not user_info:
        return ""
    return f"User roles: {user_info['roles']}, display: {user_info['display_name']}"


async def setup(bot):
    await bot.add_cog(Chat(bot))
