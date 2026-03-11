from discord.ext import commands

from config import log


async def get_server_info(guild):
    return {
        'server_name': guild.name,
        'server_id': guild.id,
        'member_count': guild.member_count,
        'members': [
            {
                'name': member.name,
                'display_name': member.display_name,
                'id': member.id,
                'bot': member.bot,
                'roles': [role.name for role in member.roles],
                'joined_at': member.joined_at,
                'top_role': member.top_role.name,
                'status': str(member.status),
            }
            for member in guild.members
        ],
    }


class Members(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_ready(self):
        log.info("Logged in as %s", self.bot.user)
        for guild in self.bot.guilds:
            self.bot.server_data[guild.id] = await get_server_info(guild)
        log.info("Loaded data for %d servers", len(self.bot.server_data))

    @commands.Cog.listener()
    async def on_member_update(self, before, after):
        guild_id = after.guild.id
        if guild_id not in self.bot.server_data:
            return
        updated_members = [
            {**member, 'status': str(after.status)} if member['id'] == after.id else member
            for member in self.bot.server_data[guild_id]['members']
        ]
        self.bot.server_data[guild_id] = {
            **self.bot.server_data[guild_id],
            'members': updated_members,
        }


async def setup(bot):
    await bot.add_cog(Members(bot))
