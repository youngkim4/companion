import logging
import sys
from dotenv import load_dotenv
import os

load_dotenv()

MAX_PROMPT_LENGTH = 2000
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

BOT_PERSONALITY = """
You are an companion assistant bot that helps users with tasks. You have a personality of a mentor, tutor, coach, therapist, etc.
You are friendly, non-hostile, but also straight to the point and efficient. These tasks you are helping with include but
are not limited to work related topics, school/academic related topics, advice related topics, and whatever the user
needs assitance with. You will use phrases such as "have you tried ____," "maybe this will work," and more helpful 'companion'
type language.
"""

REQUIRED_ENV_VARS = {
    'discord_token': 'Discord bot token',
    'openai_key': 'OpenAI API key',
    'elevenlabs_api_key': 'ElevenLabs API key',
}

log = logging.getLogger("companion")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def get_env(key):
    value = os.getenv(key)
    if not value:
        log.error("Missing required env var: %s (%s)", key, REQUIRED_ENV_VARS.get(key, ""))
        sys.exit(1)
    return value


DISCORD_TOKEN = get_env('discord_token')
OPENAI_KEY = get_env('openai_key')
ELEVENLABS_KEY = get_env('elevenlabs_api_key')
