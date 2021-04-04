from django.contrib import admin
from django.urls import path
from medhelp import views

#Django admin customization
admin.site.site_header = "Login to Cvk007"
admin.site.site_title = "Welcome to Cvk007's Dashboard"
admin.site.index_title = "Welcome folk"

urlpatterns = [
    path('', views.index, name='index'),
    path('response', views.response, name='index'),
    path('c_result', views.c_result, name='c_result'),
    path('cancer', views.cancer, name='cancer'),
    path('d_result', views.d_result, name='d_result'),
    path('diabetes', views.diabetes, name='diabetes'),
    path('h_result', views.h_result, name='h_result'),
    path('heart', views.heart, name='heart'),

]