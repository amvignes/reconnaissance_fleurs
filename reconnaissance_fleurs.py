import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models # type: ignore
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore




# Définir les chemins de dossiers d'entraînement, de validation et de test
train_dir = '/Users/amelievignes/Downloads/PROJET PERSO/DATA/train'
validation_dir = '/Users/amelievignes/Downloads/PROJET PERSO/DATA/validation'
test_dir = '/Users/amelievignes/Downloads/PROJET PERSO/DATA/test'

# Générer les images et les normaliser
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Générer les images et étiquettes pour chaque ensemble - défini les trucs
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'  # Pour classification multi-classes avec des labels numérotés
)

#'rose': 0, 'tournesol': 1, 'tulipe': 2

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'
)




# Créer un modèle CNN
model = models.Sequential()

# Ajouter des couches convolutives et de pooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Ajouter des couches fully connected (couches denses)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))  # 3 classes (rose, tulipe, tournesol)

# Afficher le résumé du modèle
model.summary()





# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# Entraîner le modèle
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=20,  # Nombre d'époques ajustable
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)


# Évaluer le modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // 32)
print(f"Précision sur l'ensemble de test : {test_acc}")



# Charger une nouvelle image
img_path = '/Users/amelievignes/Downloads/PROJET PERSO/trn.jpeg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
img_array /= 255.  # Normaliser

# Prédire la classe
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
print(f"Classe prédite : {predicted_class}")


# Sauvegarder le modèle
model.save('mon_modele_fleurs.h5')
