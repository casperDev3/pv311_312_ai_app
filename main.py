from os import getenv
from aiogram import Bot, Dispatcher, types, exceptions
from aiogram.filters import Command
from aiogram.types import Message
import logging
import asyncio
from utils import analyze_sentiment, COMMANDS_TEXTS

# Налаштування
TOKEN = getenv("BOT_TOKEN")
logging.basicConfig(level=logging.INFO)

# Ініціалізація бота
dp = Dispatcher()

# Command handler
@dp.message(Command("start"))
async def command_start_handler(message: Message) -> None:
    await message.answer(COMMANDS_TEXTS["welcome"], parse_mode='HTML')

@dp.message(Command("help"))
async def command_start_handler(message: Message) -> None:
    await message.answer(COMMANDS_TEXTS["welcome"], parse_mode='HTML')

@dp.message(Command("sentiment"))
async def command_sentiment(message: Message) -> None:
    response = analyze_sentiment("bad day(( worse whathes)) windly ... shit")
    print("__resp", response)
    await message.answer("Done!")

# Run the bot
async def main() -> None:
    bot = Bot(token=TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
