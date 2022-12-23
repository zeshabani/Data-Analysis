from django.shortcuts import render
from .plots import my_histogram, scatter
from django.http import JsonResponse

def get_stat(request):
    k = request.GET.get("k", 1)
    k = int(k) if k else 1
    result2 = scatter(k)
    scatterChartData = []
    for res in result2:
        scatterChartData += [{"customergroup":res[0], "AveR":res[1], "AveF":res[2], "AveM":res[3]}]

    return JsonResponse({
        'scatterChartData': scatterChartData
    })


def index(request):
    context = {}
    my_histogram()
    # scatter(2)
    return render(request, "index.html", context)





