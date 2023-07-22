from django.shortcuts import render, redirect
from django.http import JsonResponse

from waterApp.ai.water_ai import WaterAnalysis

from django.http import HttpResponse



# Create your views here.
def home(request):
    return render(request, 'waterApp/home.html')


def your_submission_url(request):
    main = WaterAnalysis()

    if request.method == 'POST':
        feature_1 = float(request.POST.get('feature_1'))
        feature_2 = float(request.POST.get('feature_2'))
        feature_3 = float(request.POST.get('feature_3'))
        feature_4 = float(request.POST.get('feature_4'))

        print(f"Feature 1: {feature_1}, Feature 2: {feature_2}, Feature 3: {feature_3}, Feature 4: {feature_4}")

        result = main.water_predict(feature_1, feature_2, feature_3, feature_4)

        context = {
            'pred' : result
        }

        return render(request, 'waterApp/water_result.html', context)
    else:
        pass

    return render(request, 'waterApp/water_result.html', {})




# def your_submission_url(request):
#     return HttpResponse("Your submission is being processed.")