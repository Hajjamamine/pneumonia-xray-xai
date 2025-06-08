from django.urls import path
from .views import PredictView, ExplainView  # Import the correct view

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('shap/', ExplainView.as_view(), name='shap'),  # Use ExplainView for SHAP explanations
]