import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50 # Machine Learning Algorithm
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    'D:\\Files\\Projects\\PLDI\\Code\\Potato\\Train',  # Update with the full path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight'])

valid_generator = valid_datagen.flow_from_directory(
    'D:\\Files\\Projects\\PLDI\\Code\\Potato\\Valid',  # Update with the full path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight'])

# Load ResNet50 pre-trained model without top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
predictions = Dense(3, activation='softmax')(x)  # 3 classes

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze some layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks for model training
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with callbacks
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20,  # Increase the number of epochs for better convergence
    callbacks=[checkpoint, early_stopping])

# Save the trained model
model.save('potato_leaf.h5')
