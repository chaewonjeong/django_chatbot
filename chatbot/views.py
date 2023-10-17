from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import PreTrainedTokenizerFast
from django.core.cache import cache
from transformers import GPT2LMHeadModel
import torch

# 토큰 정의
Q_TKN = "<usr>"  # 사용자 질문 토큰
A_TKN = "<sys>"  # 시스템 답변 토큰
BOS = "</s>"  # 문장 시작 토큰
EOS = "</s>"  # 문장 종료 토큰
MASK = "<unused0>"  # 마스크 토큰
SENT = "<unused1>"  # 미사용 토큰
PAD = "<pad>"  # 패딩 토큰

# 허깅페이스 transformers 에 등록된 사전 학습된 koGTP2 토크나이저를 가져온다.
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token=BOS,
    eos_token=EOS,
    unk_token="<unk>",
    pad_token=PAD,
    mask_token=MASK,
)


def home(request):
    context = {}
    return render(request, "home.html", context)


# 전체 모델이 저장된 경우 사용
def load_model():
    device = torch.device("cpu")
    model_path = "/Users/jeongchaewon/workspace_local/Dev/django_chatbot/chatbot/static/model/model.pt"
    model = cache.get("cached_model")

    if model is None:
        try:
            model = torch.load(model_path, map_location=device)
            cache.set("cached_model", model)
        except Exception as e:
            return None
    return model


# state_dict만 저장된 파일 불러올 경우
def load_model_weight():
    device = torch.device("cpu")
    model_path = "/Users/jeongchaewon/workspace_local/Dev/django_chatbot/chatbot/static/model/model_e5_b32.pth"
    model = cache.get("cached_model")

    if model is None:
        try:
            model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
            model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            cache.set("cached_model", model)
        except Exception as e:
            return None
    return model


def answer(q, model, tokenizer):
    with torch.no_grad():
        q = q.strip()
        a = ""
        while 1:
            input_ids = (
                torch.LongTensor(tokenizer.encode(Q_TKN + q + SENT + A_TKN + a))
                .cpu()
                .unsqueeze(dim=0)
            )
            pred = model(input_ids)
            pred = pred.logits.cpu()
            gen = tokenizer.convert_ids_to_tokens(
                torch.argmax(pred, dim=-1).squeeze().numpy().tolist()
            )[-1]
            if gen == EOS:
                break
            a += gen.replace("▁", " ")
        return a


@csrf_exempt
def chatanswer(request):
    context = {}

    chattext = request.GET["chattext"]

    # model = load_model()
    model = load_model()

    if model is None:
        return JsonResponse({"error": "모델 로드 오류"})

    a = answer(chattext, model, koGPT2_TOKENIZER)

    context["anstext"] = a.strip()

    return JsonResponse(context, content_type="application/json")
