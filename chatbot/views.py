import json
import os
import torch
from django.shortcuts import render
from django.core.cache import cache
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast


def home(request):
    context = {}
    return render(request, "home.html", context)


def chatanswer(request):
    context = {}

    chattext = request.GET["chattext"]

    return JsonResponse(context, content_type="application/json")

    return JsonResponse(context, content_type="application/json")
