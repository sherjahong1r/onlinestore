import os
from kaggle.api.kaggle_api_extended import KaggleApi

# API ni ishga tushirish
api = KaggleApi()
api.authenticate()

# Datasetni yuklab olish
dataset_name = 'inaturalist-2021'
print(f"{dataset_name} yuklanmoqda...")

# Bu kod datasetni joriy papkaga yuklaydi
api.competition_download_files(dataset_name, path='./data')
print("Yuklab olindi! './data' papkasini tekshiring.")




import tensorflow as tf

def get_data_generator(data_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Rasmlarni batch (bo'lak) ko'rinishida o'qish
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    return generator

# Ishlatilishi:
# train_gen = get_data_generator('./data/train/Plantae')




import tensorflow as tf
print("GPU mavjud:", tf.config.list_physical_devices('GPU'))

















