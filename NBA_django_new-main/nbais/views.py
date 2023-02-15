from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'nbais/index.html')

def signin(request):
    return render(request, 'nbais/signin.html')

def signup(request):
    return render(request, 'nbais/signup.html')

def index2(request):
    return render(request, 'nbais/index.html')
