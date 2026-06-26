# import asyncio
# import os
# import torch
# import librosa
# from aiogram import Bot, Dispatcher, types, F
# from transformers import WhisperForConditionalGeneration, WhisperProcessor
# from pydub import AudioSegment

# # Model path
# # model_path = model_path = r"C:\Users\Owner\Desktop\Text_To_Speach\my_whisper_uz_model\my_whisper_uz_model"
# model_path = model_path = r"C:\Users\Owner\Desktop\Text_To_Speach\whisper-medium-uzbek-lora-final"
# # Model mavjudligini tekshirish
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model topilmadi: {model_path}")

# # Modelni yuklash
# print("Model yuklanmoqda...")
# processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
# model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
# model.eval()
# print("Model yuklandi!")

# # Token
# TOKEN = "8826396186:AAE0smRv4k2QmOwn5wUsoJwEC_JXoZcH724"
# bot = Bot(token=TOKEN)
# dp = Dispatcher()

# @dp.message(F.voice)
# async def handle_voice(message: types.Message):
#     try:
#         await message.answer("🎙 Ovoz tahlil qilinmoqda...")
        
#         # Faylni yuklash
#         file = await bot.get_file(message.voice.file_id)
#         await bot.download_file(file.file_path, "voice.ogg")
        
#         # OGG → WAV
#         audio_seg = AudioSegment.from_ogg("voice.ogg")
#         audio_seg.export("voice.wav", format="wav")

#         # Ovozni o'qish va modelga uzatish
#         audio, sr = librosa.load("voice.wav", sr=16000)
#         inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        
#         # with torch.no_grad():
#             # generated_ids = model.generate(inputs["input_features"])
#         with torch.no_grad():
#          generated_ids = model.generate(
#          inputs["input_features"],
#          language="uz",
#          task="transcribe"
#     )



#         result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
#         if result.strip():
#             await message.reply(f"📝 Siz aytgan matn:\n{result}")
#         else:
#             await message.reply("❌ Matn aniqlanmadi.")
            
#     except Exception as e:
#         await message.reply(f"⚠️ Xatolik: {e}")
    
#     finally:
#         # Vaqtinchalik fayllarni o'chirish
#         for f in ["voice.ogg", "voice.wav"]:
#             if os.path.exists(f):
#                 os.remove(f)

# @dp.message(F.text)
# async def handle_text(message: types.Message):
#     await message.reply("🎤 Menga ovozli xabar yuboring, matniga o'girib beraman!")

# async def main():
#     print("✅ Bot ishga tushdi...")
#     await dp.start_polling(bot)

# if __name__ == '__main__':
#     asyncio.run(main())


import asyncio
import os
import torch
import librosa
from aiogram import Bot, Dispatcher, types, F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from pydub import AudioSegment
from peft import PeftModel

# ⭐ MODEL PAPKASI

model_path = r"C:\Users\Owner\Desktop\Text_To_Speach\whisper-medium-uzbek-lora-final"
# whisper-medium-uzbek-lora-final
# ⭐ Papka mavjudligini tekshirish
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model papkasi topilmadi: {model_path}")

print("✅ Model papkasi tayyor!")

# ⭐ 1. Asosiy modelni yuklash
base_model_name = "openai/whisper-medium"
print(f"🔄 Asosiy model yuklanmoqda: {base_model_name}")

processor = WhisperProcessor.from_pretrained(base_model_name)
base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)

print("✅ Asosiy model yuklandi!")

# ⭐ 2. LoRA adapterni yuklash
print("🔄 LoRA adapter yuklanmoqda...")
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

print("✅ Model yuklandi!")

TOKEN = "8826396186:AAE0smRv4k2QmOwn5wUsoJwEC_JXoZcH724"
bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(F.voice)
async def handle_voice(message: types.Message):
    try:
        await message.answer("🎙 Ovoz tahlil qilinmoqda...")
        
        file = await bot.get_file(message.voice.file_id)
        await bot.download_file(file.file_path, "voice.ogg")
        
        audio_seg = AudioSegment.from_ogg("voice.ogg")
        audio_seg.export("voice.wav", format="wav")

        audio, sr = librosa.load("voice.wav", sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_features"],
                language="uz",
                task="transcribe"
            )

        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if result.strip():
            await message.reply(f"📝 Siz aytgan matn:\n{result}")
        else:
            await message.reply("❌ Matn aniqlanmadi.")
            
    except Exception as e:
        await message.reply(f"⚠️ Xatolik: {e}")
    
    finally:
        for f in ["voice.ogg", "voice.wav"]:
            if os.path.exists(f):
                os.remove(f)

@dp.message(F.text)
async def handle_text(message: types.Message):
    await message.reply("🎤 Menga ovozli xabar yuboring, matniga o'girib beraman!")

async def main():
    print("✅ Bot ishga tushdi...")
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())