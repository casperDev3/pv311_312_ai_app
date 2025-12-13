# !pip install unsloth transformers datasets

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset


max_seq_length = 2048 # Максимальна довжина тексту
dtype = None # Автовизначення (Float16 або Bfloat16)
load_in_4bit = True # Стиснення, щоб влізло в пам'ять

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", # Базова модель
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Ступінь адаптації (чим більше, тим розумніше, але важче)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)


alpaca_prompt = """### Питання: Якого кольору небо?
### Відповідь: Зеленого."""
data_dict = {
    'text': [alpaca_prompt]
}
dataset = Dataset.from_dict(data_dict)

print(dataset)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset, # Ваш підготовлений набір даних
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Для демо-пари ставимо мало кроків
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

trainer.train() # ЗАПУСК МАГІЇ

FastLanguageModel.for_inference(model) # Оптимізація для відповідей

inputs = tokenizer(
[
    "Питання: Як пройти до моря?"
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64)
print(tokenizer.batch_decode(outputs))