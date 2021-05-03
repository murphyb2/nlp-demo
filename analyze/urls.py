from django.urls import path

from . import views
app_name = 'analyze'
urlpatterns = [
    path('', views.index, name='index'),
]
