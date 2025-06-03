import os 
import tensorflow as tf
from django.conf import settings

# Load model (only once when the server starts)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'my_real_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import tempfile
from rest_framework.parsers import MultiPartParser

def preprocess(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

class PredictView(APIView):
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