import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, last_conv_layer_name):
        """
        Initialize the Grad-CAM class.
        
        Args:
            model: Trained Keras model.
            last_conv_layer_name: Name of the last convolutional layer in the model.
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self.grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

    def compute_heatmap(self, img_array, pred_index=None):
        """
        Compute Grad-CAM heatmap for a given image.
        
        Args:
            img_array: Preprocessed input image as a NumPy array.
            pred_index: Index of the predicted class (optional).
        
        Returns:
            Heatmap as a NumPy array.
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap

    def overlay_heatmap(self, heatmap, img, alpha=0.4, cmap='viridis'):
        """
        Overlay the Grad-CAM heatmap on the original image.
        
        Args:
            heatmap: Grad-CAM heatmap as a NumPy array.
            img: Original image as a NumPy array.
            alpha: Transparency factor for the heatmap overlay.
            cmap: Colormap for the heatmap.
        
        Returns:
            Overlayed image as a NumPy array.
        """
        heatmap = np.uint8(255 * heatmap)
        colormap = plt.cm.get_cmap(cmap)
        colormap = colormap(np.arange(256))[:, :3]
        colormap = colormap[heatmap]
        colormap = np.uint8(colormap * 255)

        overlayed_img = colormap * alpha + img
        overlayed_img = np.clip(overlayed_img, 0, 255).astype('uint8')
        return overlayed_img