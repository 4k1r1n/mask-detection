from django.urls import path
from project import views

urlpatterns = [
    path('', views.button),
    path('external/', views.external),
]
