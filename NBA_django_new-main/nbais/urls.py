from django.urls import path
from .views import * 

app_name = 'nbais'
urlpatterns = [
    path('', index, name='index'),
    path('signin.html', signin, name='signin'),
    path('signup.html',signup, name='signup'),
    path('index.html', index2, name='index2'),
]