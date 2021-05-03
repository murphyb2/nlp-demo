from django.shortcuts import render
from .spNLP import nlp

# Create your views here.
def index(request):
    data = nlp()
    context = {'data': data}
    print(data)
    return render(request, 'index.html', context)