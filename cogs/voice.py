from io import BytesIO

import discord
import ffmpeg
from discord.ext import commands
from elevenlabs.client import ElevenLabs
from openai import AsyncOpenAI
from pydub import AudioSegment

from config import DEFAULT_VOICE_ID, ELEVENLABS_KEY, OPENAI_KEY, log

openai_client = AsyncOpenAI(api_key=OPENAI_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_KEY)


def synthesize_speech(text, voice_id=DEFAULT_VOICE_ID):
    try:
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
        )
        return audio
    except Exception as e:
        log.error("Failed to synthesize speech: %s", e)
        return None


async def transcribe_speech(audio_source):
    audio_data = BytesIO()
    audio_data.write(audio_source.read())
    audio_data.seek(0)
    audio_data.name = "audio.wav"
    try:
        transcript = await openai_client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_data,
        )
        return transcript.text
    except Exception as e:
        log.error("Failed to transcribe speech: %s", e)
        return None


def process_audio(audio_source):
    sound = AudioSegment.from_raw(
        audio_source, sample_width=2, frame_rate=48000, channels=1
    )
    buffer = BytesIO()
    sound.export(buffer, format="wav")
    buffer.seek(0)
    return buffer


async def play_speech(vc, source_path):
    process = (
        ffmpeg
        .input(source_path)
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=2, ar='48k')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    audio_source = discord.PCMAudio(process.stdout)
    vc.play(audio_source)


class Voice(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def join(self, ctx):
        if not ctx.author.voice:
            await ctx.send("You need to be in a voice channel first.")
            return
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send("I've joined the voice channel!")

    @commands.command()
    async def leave(self, ctx):
        if ctx.guild.voice_client:
            await ctx.guild.voice_client.disconnect()
            await ctx.send("I've left the voice channel!")
        else:
            await ctx.send("I'm not in a voice channel.")


async def setup(bot):
    await bot.add_cog(Voice(bot))
