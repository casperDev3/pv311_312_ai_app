from aiogram import types
from aiogram.fsm.context import FSMContext
from states.prompts import RegistrationStates
from main import dp
from utils import analyze_sentiment


@dp.message(RegistrationStates.wait_for_prompt)
async def get_prompt(msg: types.Message, state: FSMContext):
    text = msg.text

    # Analyze the sentiment of user's message
    response = analyze_sentiment(text)
    await msg.answer(f"Sentiment analysis result: {response}")

    await state.clear()