import logging
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from transformers import pipeline

# 1. Botni sozlash
API_TOKEN = 'YOUR_API_TOKEN'
logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# 2. O'z modelingizni yuklash
# 'model' qismiga papkangiz nomini yozing
pipe = pipeline("automatic-speech-recognition", model="./my_whisper_uz_model")

@dp.message_handler(content_types=['voice'])
async def handle_voice(message: types.Message):
    await message.answer("Ovoz qabul qilindi, matnga o'girilmoqda...")
    
    # Telegramdan faylni yuklab olish
    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "voice.ogg")
    
    # 3. Model orqali matnga o'girish
    result = pipe("voice.ogg")
    
    # Natijani userga yuborish
    await message.reply(f"Matn: {result['text']}")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)


    