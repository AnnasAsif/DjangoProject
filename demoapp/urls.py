from django.urls import path 
from . import views 
 
urlpatterns = [ 
    path('api/imageMerge', views.imageMerge),
    path('api/imagePartRemove', views.imagePartRemove),
    path('api/imageDetectObject', views.imageDetectObject),
    path('api/removeObject', views.removeObject),
    # path('api/tutorials/:pk', views.tutorial_detail),
    # path('api/tutorials/published', views.tutorial_list_published)
]