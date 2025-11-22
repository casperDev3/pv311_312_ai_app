from os import getenv
from aiogram import Bot, Dispatcher, types, exceptions
from aiogram.filters import Command
from aiogram.types import Message
import logging
import asyncio
from utils import analyze_sentiment, COMMANDS_TEXTS
from aiogram.fsm.context import FSMContext
from states.prompts import RegistrationStates

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
async def command_help_handler(message: Message) -> None:  # Fixed duplicate function name
    await message.answer(COMMANDS_TEXTS["welcome"], parse_mode='HTML')


@dp.message(Command("sentiment"))
async def command_sentiment(message: Message, state: FSMContext) -> None:
    response = analyze_sentiment("bad day(( worse whathes)) windly ... shit")
    await state.set_state(RegistrationStates.wait_for_prompt)
    print("__resp", response)
    await message.answer("Please send me your prompt for analysis:")


# Handler for state - moved here to avoid circular imports
@dp.message(RegistrationStates.wait_for_prompt)
async def get_prompt(msg: Message, state: FSMContext):
    print("State handler triggered")
    text = msg.text
    print("__text", text)

    # Analyze the sentiment of user's message
    response = analyze_sentiment(text)
    await msg.answer(f"Sentiment analysis result: {response}")

    await state.clear()


# Run the bot
async def main() -> None:
    bot = Bot(token=TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())