from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name = 'index'),
    path('get_stat', views.get_stat, name='get_stat'),
]