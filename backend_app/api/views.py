import os
import tensorflow as tf
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import tempfile
from rest_framework.parsers import MultiPartParser
import cv2
import numpy as np
import shap  # Import SHAP
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import base64  # Import base64 for encoding images
from typing import Tuple, Dict  # Import Tuple and Dict for type hints

# Load model (only once when the server starts)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'my_real_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(image_path):
    """
    Preprocess the input image for the model.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

class PredictView(APIView):
    """
    Endpoint for making predictions on X-ray images.
    """
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        file_obj = request.FILES.get('image')

        if not file_obj:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg') as temp_file:
            for chunk in file_obj.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        try:
            image = preprocess(temp_path)
            predictions = model.predict(image)
            label = "Normal" if predictions[0][0] < 0.5 else "Pneumonia"
            return Response({
                "prediction": label,
                "confidence": float(predictions[0][0]),
            })
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)    
        finally:
            os.remove(temp_path)  # Clean up the temporary file

class PneumoniaSHAPExplainer:
    def __init__(self, model, class_names: Tuple[str, str] = ("Normal", "Pneumonia")):
        """
        Initialize the SHAP explainer for pneumonia detection.
        
        Args:
            model: Your trained Keras model
            class_names: Tuple of (negative_class, positive_class) names
        """
        self.model = model
        self.class_names = class_names
        
        # Initialize with empty masker (will be created on first explanation)
        self.masker = None
        self.explainer = None
        
        # Pre-allocate memory for better performance
        self.image_shape = (224, 224, 3)
        self.batch_size = 32
        
    def _initialize_explainer(self, example_image: np.ndarray):
        """Initialize the SHAP explainer on first use"""
        self.masker = shap.maskers.Image("inpaint_telea", example_image.shape)
        
        def predict_fn(x):
            x_preprocessed = self._preprocess_shap_batch(x)
            predictions = self.model(x_preprocessed).numpy()  # Ensure predictions are numpy arrays
            return np.concatenate([1 - predictions, predictions], axis=1)  # Binary classification
    
        self.explainer = shap.Explainer(
            predict_fn,
            self.masker,
            output_names=self.class_names,
            seed=42
        )
    
    def _preprocess_shap_batch(self, x: np.ndarray) -> tf.Tensor:
        """Batch preprocessing for SHAP input"""
        x = tf.convert_to_tensor(x)
        if x.shape[-1] == 1:
            x = tf.image.grayscale_to_rgb(x)
        x = tf.image.resize(x, self.image_shape[:2])
        return tf.cast(x, tf.float32) / 255.0
    
    def preprocess_single_image(self, image_path: str) -> tf.Tensor:
        """Preprocess a single image for model prediction"""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        if image.shape[-1] == 1:
            image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, self.image_shape[:2])
        return tf.expand_dims(tf.cast(image, tf.float32) / 255.0, 0)
    
    def explain_image(self, image_path: str, max_evals: int = 200) -> Dict:
        """
        Generate SHAP explanation for a single image.

        Args:
            image_path: Path to the image file
            max_evals: Number of evaluations for SHAP (higher=more accurate)

        Returns:
            Dictionary containing:
            - prediction: class prediction
            - confidence: prediction probability
            - shap_values: raw SHAP values
            - heatmap: base64-encoded heatmap for visualization
        """
        try:
            # Load and preprocess image
            image_np = np.array(tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3))
            if image_np.shape[-1] == 1:
                image_np = np.repeat(image_np, 3, axis=-1)
            
            # Initialize explainer on first run
            if self.explainer is None:
                self._initialize_explainer(image_np)
            
            # Get model prediction
            img_tensor = self.preprocess_single_image(image_path)
            pred_prob = float(self.model.predict(img_tensor)[0][0])
            prediction = self.class_names[1] if pred_prob >= 0.5 else self.class_names[0]
            
            # Generate SHAP values
            shap_values = self.explainer(
                np.expand_dims(image_np, axis=0),
                max_evals=max_evals,
                batch_size=self.batch_size
            )
            
            # Ensure shap_values is structured correctly
            if not hasattr(shap_values, 'values') or shap_values.values is None:
                raise ValueError("SHAP values could not be computed.")
            
            # Create heatmap visualization
            plt.figure(figsize=(4, 4))  # Reduce figure size
            shap.image_plot(
                shap_values,
                -np.expand_dims(image_np, axis=0),
                show=False
            )
            
            # Encode the plot as a base64 string
            from io import BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=50)  # Reduce DPI for smaller size
            buffer.seek(0)
            heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close()
            
            return {
                "prediction": prediction,
                "confidence": pred_prob if prediction == self.class_names[1] else 1 - pred_prob,
                "heatmap": heatmap_base64,  # Return base64-encoded heatmap
                "shap_summary": shap_values.values[0].tolist()[:10]  # Send only the top 10 SHAP values
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "prediction": "Error",
                "confidence": 0.0
            }

class ExplainView(APIView):
    """
    Endpoint for generating SHAP explanations for X-ray images.
    """
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        file_obj = request.FILES.get('image')

        if not file_obj:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg') as temp_file:
            for chunk in file_obj.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        try:
            explainer = PneumoniaSHAPExplainer(model)
            explanation = explainer.explain_image(temp_path)
            return Response(explanation)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            os.remove(temp_path)  # Clean up the temporary file
