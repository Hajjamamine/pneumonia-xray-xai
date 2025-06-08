import tensorflow as tf

model_path = "c:/Users/PC/Desktop/Projets/pneumonia_xray_xai/backend_app/model/my_real_model.keras"
model = tf.keras.models.load_model(model_path)

# Get the resnet50 base model
resnet_layer = model.get_layer("resnet50")

# Print all inner layer names to find conv layers
for layer in resnet_layer.layers[::-1]:
    if 'conv' in layer.name:
        #print(f"Last conv layer inside ResNet50: {layer.name}")
        break
#model.get_layer('resnet50').summary()
model.summary()