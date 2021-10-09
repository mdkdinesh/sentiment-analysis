from django.urls import path
from . import views

urlpatterns = [
    path("", views.welcome),
    path("index/", views.index),
    path('dataset/', views.dataset),
    path("chart/", views.charts_sample),
    path("freq/", views.freq),
    path('total_dataset/', views.total_dataset),
    path('welcome/', views.welcome),
]
