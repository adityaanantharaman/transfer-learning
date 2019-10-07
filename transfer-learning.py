# coding: utf-8

# In[1]:
import pandas as pd
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

# In[2]:


base_model = MobileNet(weights='imagenet',
                       include_top=False)  #imports the mobilenet model and discards the last 1000 neuron layer.

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(
    x)  #we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation='relu')(x)  #dense layer 2
x = Dense(512, activation='relu')(x)  #dense layer 3
preds = Dense(3, activation='softmax')(x)  #final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)

for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = datagen.flow_from_directory(
    './data/train/',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=8,
    class_mode='categorical',
    shuffle=True
)
val_generator = datagen.flow_from_directory(
    './data/val/',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=8,
    class_mode='categorical',
    shuffle=True
)
test_generator = datagen.flow_from_directory(
    './data/test/',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=8,
    class_mode='categorical',
    shuffle=True
)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
step_size_train = train_generator.n // train_generator.batch_size

model.fit_generator(
    generator=train_generator,
    validation_data=val_generator,
    steps_per_epoch=step_size_train,
    epochs=5
)

score = model.evaluate_generator(test_generator)
df = pd.DataFrame().assign(test_loss=[score[0]], test_accuracy=[score[1]]).to_csv('./output/test_results.csv')
model.save_weights('./output/my_model_weights.h5')
