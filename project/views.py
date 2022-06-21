from django.shortcuts import render
import requests
import sys
from subprocess import run, PIPE
from django.core.files.storage import FileSystemStorage
#from image import process_image
from maskDetection.process_image import process_image


def button(request):
    return render(request, 'index.html')


def external(request):
    image = request.FILES['image']
    file_name = FileSystemStorage().save(image.name, image)
    file_url = FileSystemStorage().open(file_name)

    template_url = FileSystemStorage().url(file_name)

    image = process_image(str(file_url), str(file_name))
    edit_url = FileSystemStorage().url(image)

    return render(request, 'index.html', {'raw_url': template_url, 'edit_url': edit_url})
