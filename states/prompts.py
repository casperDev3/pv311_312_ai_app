from aiogram.fsm.state import State, StatesGroup


class RegistrationStates(StatesGroup):
    wait_for_prompt = State()
