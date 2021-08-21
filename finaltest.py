# 題目
你要訓練出能夠辨識新鮮與腐壞水果的新模型。
你必須讓模型的驗證準確度達到 92%才能通過測驗
但我們期望你可以盡力拿出更好的表現。你可以運用在前幾個練習中學到的技能。
具體地說，我們建議使用一些結合遷移學習、資料增強和微調的方式。
將模型訓練到在驗證資料集上有至少 92% 的準確度之後，請先儲存模型，然後再評估模型的準確度。
現在就開始吧！

#限制
在這個練習中，你要訓練出可以辨識新鮮與腐壞水果的新模型。
資料集來自 Kaggle，如果你在本課程結束後有興趣展開新專案，這裡有絕佳的資源。
資料集結構位於 fruits資料夾。
其中共有六種類型的水果：新鮮蘋果、新鮮柳橙、新鮮香蕉、腐壞蘋果、腐壞柳橙和腐壞香蕉。
這表示模型的輸出層必須有 6 個神經元，才能成功完成分類。
你也需要運用 categorical_crossentropy來編寫模型，因為我們的類型超過兩種。


from tensorflow import keras

base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False)

# Freeze base model
base_model.trainable = False
# Create inputs with correct shape
inputs = keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

# Add pooling layer or flatten layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Add final dense layer
outputs = keras.layers.Dense(6, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)
model.summary()
model.compile(loss = keras.losses.BinaryCrossentropy(from_logits=True) , metrics = [keras.metrics.BinaryAccuracy()])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)
# load and iterate training dataset
train_it = datagen.flow_from_directory('fruits/train/', 
                                       target_size=(224, 224), 
                                       color_mode='rgb', 
                                       class_mode="categorical")
# load and iterate validation dataset
valid_it = datagen.flow_from_directory('fruits/valid/', 
                                      target_size=(224, 224), 
                                      color_mode='rgb', 
                                      class_mode="categorical")
model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=15)
model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)

執行結果:
Epoch 1/15
37/36 [==============================] - 28s 751ms/step - loss: 0.7148 - binary_accuracy: 0.8281 - val_loss: 0.6905 - val_binary_accuracy: 0.8784
Epoch 2/15
37/36 [==============================] - 19s 513ms/step - loss: 0.6904 - binary_accuracy: 0.8782 - val_loss: 0.6817 - val_binary_accuracy: 0.8977
Epoch 3/15
37/36 [==============================] - 19s 511ms/step - loss: 0.6844 - binary_accuracy: 0.8904 - val_loss: 0.6807 - val_binary_accuracy: 0.8987
Epoch 4/15
37/36 [==============================] - 19s 513ms/step - loss: 0.6824 - binary_accuracy: 0.8952 - val_loss: 0.6804 - val_binary_accuracy: 0.8987
Epoch 5/15
37/36 [==============================] - 19s 512ms/step - loss: 0.6735 - binary_accuracy: 0.9137 - val_loss: 0.6639 - val_binary_accuracy: 0.9357
Epoch 6/15
37/36 [==============================] - 19s 512ms/step - loss: 0.6604 - binary_accuracy: 0.9387 - val_loss: 0.6523 - val_binary_accuracy: 0.9585
Epoch 7/15
37/36 [==============================] - 19s 513ms/step - loss: 0.6471 - binary_accuracy: 0.9700 - val_loss: 0.6493 - val_binary_accuracy: 0.9645
Epoch 8/15
37/36 [==============================] - 19s 513ms/step - loss: 0.6440 - binary_accuracy: 0.9752 - val_loss: 0.6405 - val_binary_accuracy: 0.9807
Epoch 9/15
37/36 [==============================] - 19s 515ms/step - loss: 0.6383 - binary_accuracy: 0.9870 - val_loss: 0.6426 - val_binary_accuracy: 0.9767
Epoch 10/15
37/36 [==============================] - 19s 512ms/step - loss: 0.6374 - binary_accuracy: 0.9897 - val_loss: 0.6381 - val_binary_accuracy: 0.9853
Epoch 11/15
37/36 [==============================] - 19s 505ms/step - loss: 0.6356 - binary_accuracy: 0.9913 - val_loss: 0.6387 - val_binary_accuracy: 0.9843
Epoch 12/15
37/36 [==============================] - 19s 508ms/step - loss: 0.6349 - binary_accuracy: 0.9922 - val_loss: 0.6385 - val_binary_accuracy: 0.9863
Epoch 13/15
37/36 [==============================] - 19s 512ms/step - loss: 0.6347 - binary_accuracy: 0.9932 - val_loss: 0.6371 - val_binary_accuracy: 0.9883
Epoch 14/15
37/36 [==============================] - 19s 510ms/step - loss: 0.6337 - binary_accuracy: 0.9938 - val_loss: 0.6371 - val_binary_accuracy: 0.9878
Epoch 15/15
37/36 [==============================] - 19s 510ms/step - loss: 0.6334 - binary_accuracy: 0.9937 - val_loss: 0.6380 - val_binary_accuracy: 0.9848
<tensorflow.python.keras.callbacks.History at 0x7fa06c39ccf8>
