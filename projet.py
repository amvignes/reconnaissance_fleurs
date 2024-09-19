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
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Faire pivoter les images
    width_shift_range=0.2,  # Décaler horizontalement
    height_shift_range=0.2,  # Décaler verticalement
    shear_range=0.2,  # Appliquer une déformation affine
    zoom_range=0.2,  # Appliquer un zoom
    horizontal_flip=True,  # Retourner l'image horizontalement
    fill_mode='nearest'  # Remplir les pixels vides après transformation
)
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Faire pivoter les images
    width_shift_range=0.2,  # Décaler horizontalement
    height_shift_range=0.2,  # Décaler verticalement
    shear_range=0.2,  # Appliquer une déformation affine
    zoom_range=0.2,  # Appliquer un zoom
    horizontal_flip=True,  # Retourner l'image horizontalement
    fill_mode='nearest'  # Remplir les pixels vides après transformation
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Faire pivoter les images
    width_shift_range=0.2,  # Décaler horizontalement
    height_shift_range=0.2,  # Décaler verticalement
    shear_range=0.2,  # Appliquer une déformation affine
    zoom_range=0.2,  # Appliquer un zoom
    horizontal_flip=True,  # Retourner l'image horizontalement
    fill_mode='nearest'  # Remplir les pixels vides après transformation
)
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

# Ajouter des couches convolutives et de pooling : changement de couches
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
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

from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=50,  # Augmenter le nombre d'époques
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    callbacks=[early_stopping, reduce_lr])

# Évaluer le modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // 32)
print(f"Précision sur l'ensemble de test : {test_acc}")



# Charger une nouvelle image
img_path = '/Users/amelievignes/Downloads/PROJET PERSO/tlp.jpeg'
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
