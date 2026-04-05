"""
URL configuration for tulun project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, reverse_lazy
from django.contrib.auth import views as auth_views
from django.shortcuts import redirect
from django.http import HttpResponse
from translations.views import (
    translate_view,
    create_corpus_entry,
    suggest_terms,
    eval_module_view,
    download_eval_output,
)

def healthcheck(request):
    return HttpResponse("OK")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/login/', auth_views.LoginView.as_view(template_name='translations/login.html'), name='login'),
    path('accounts/logout/', auth_views.LogoutView.as_view(template_name='translations/logout.html'), name='logout'),
    path('translate/', translate_view, name='translate'),
    # redirect to translate view
    path('', lambda request: redirect(reverse_lazy('translate')), name='home'),
    path('api/corpus-entry/', create_corpus_entry, name='create_corpus_entry'),
    path('api/suggest-terms/', suggest_terms, name='suggest_terms'),
    path('eval/', eval_module_view, name='eval_module'),
    path('eval/output/<str:filename>/', download_eval_output, name='download_eval_output'),
    path('healthcheck/', healthcheck, name='healthcheck'),
]
