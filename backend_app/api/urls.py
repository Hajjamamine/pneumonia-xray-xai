from django.urls import path
from .views import PredictView, GradCAMView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('generate_grad_cam/', GradCAMView.as_view(), name='generate_grad_cam'), 
]