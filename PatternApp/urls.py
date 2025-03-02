from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("Login.html", views.Login, name="Login"),	      
               path("UserLogin", views.UserLogin, name="UserLogin"),
               path("LoadDataset", views.LoadDataset, name="LoadDataset"),
               path("TrainML", views.TrainML, name="TrainML"),
	       path("Vector", views.Vector, name="Vector"),
	       path("Predict", views.Predict, name="Predict"),	      
	       path("PredictAction", views.PredictAction, name="PredictAction"),	
]
