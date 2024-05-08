from django.http import JsonResponse, FileResponse, HttpResponseNotFound
from django.views.decorators.csrf import csrf_exempt
from submittion.models import Submittion, Score, User
from django.core.exceptions import ObjectDoesNotExist
from django.db.models import Window, F
from django.db.models.functions.window import Rank
import os

from submittion import attack

def download_file(request):
    file_path = 'README.md.zip'  # 文件在服务器上的路径
    if os.path.exists(file_path):
        fh = open(file_path, 'rb')
        response = FileResponse(fh, as_attachment=True, filename='downloaded_file.zip')
        return response
    else:
        return HttpResponseNotFound('The requested file was not found on our server.')

def top(request):
    # 在数据库里查询平均分排名前10的用户
    if request.method == 'GET':
        top_scores = Score.objects.order_by('-average')[:10]
        score_map = {score.subcode: score for score in top_scores}
        user_infos = User.objects.filter(subcode__in=score_map.keys())

        results = [
            {
                'nickname': user.usrname,
                'student_no': user.student_no,
                'score': score_map[user.subcode].average
            }
            for user in user_infos
        ]
        print(results)
        results.sort(key=lambda x: x['score'], reverse=True)
        return JsonResponse({'status': 'success', 'data': results})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

def lookup(request):
    #拿到submission code，拿到排名，计算一下排名的百分比
    if request.method == 'GET':
        subcode = request.GET.get('subcode')
        try:
            score = Score.objects.get(subcode=subcode)
        except ObjectDoesNotExist:
            return JsonResponse({'error': 'Student not found'}, status=404)
        
        score_cnt = Score.objects.count()

        scores = Score.objects.all().order_by('-attack').iterator()
        atk_rank = 1
        for s in scores:
            if s.subcode == subcode:
                break
            atk_rank += 1
        
        scores = Score.objects.all().order_by('-defend').iterator()
        def_rank = 1
        for s in scores:
            if s.subcode == subcode:
                break
            def_rank += 1

        scores = Score.objects.all().order_by('-average').iterator()
        avg_rank = 1
        for s in scores:
            if s.subcode == subcode:
                break
            avg_rank += 1
        
        # print(atk_rank, def_rank, avg_rank, score_cnt, '%.2f' % (100 * atk_rank / score_cnt))
        # 返回前端页面
        return JsonResponse({'status': 'success',
                             'attack': "%.2f" % score.attack,
                             'defend': "%.2f" % score.defend,
                             'average': "%.2f" % score.average,
                             'attack_rank': atk_rank,
                             'defend_rank': def_rank,
                             'average_rank': avg_rank,
                             'attack_per': '%.2f' % (100 * atk_rank / score_cnt),
                             'defend_per': '%.2f' % (100 * def_rank / score_cnt),
                             'average_per': '%.2f' % (100 * avg_rank / score_cnt),})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

@csrf_exempt
def submit(request):
    # 接收文件 receive files
    if request.method == 'POST':
        py_file = request.FILES.get('pyFile')
        pt_file = request.FILES.get('ptFile')
        description = request.POST.get('description')

        if not all([py_file, pt_file, description]):
            return JsonResponse({'status': 'error', 'message': 'Missing files or description'})

        # 保存文件到磁盘 Save files to disk
        def save_file(f, subfolder, fname):
            base_dir = 'data'
            file_path = os.path.join(base_dir, subfolder, fname)
            os.system('rm %s' % file_path)
            with open(file_path, 'wb+') as destination:
                for chunk in f.chunks():
                    destination.write(chunk)
            return file_path
        
        py_file_path = save_file(py_file, 'codes', 'code' + description + '.py')
        pt_file_path = save_file(pt_file, 'models', description + '.pt')

        # 尝试评分 Attempt to rate
        try:            
            attack_socre = 0
            defend_score = 0
            submittions = Submittion.objects.all()[:10]
            for s in submittions:
                a, _ = attack('code' + description, s.subcode)
                _, d = attack('code' + s.subcode, description)
                attack_socre += a
                defend_score += d
                print(attack_socre, defend_score)
            attack_socre /= 10
            defend_score /= 10
            avgscore = (attack_socre + defend_score) / 2
            print(attack('code' + description, description))
        except:
            os.system('rm %s' % py_file_path)
            os.system('rm %s' % pt_file_path)
            return JsonResponse({'status': 'error', 'message': 'Invalid code or model'})

        # 保存记录到数据库 Save Records to Database
        _, created = Score.objects.update_or_create(
            subcode=description,
            defaults={'attack': attack_socre, 'defend': defend_score, 'average': avgscore}
        )

        _, created = Submittion.objects.update_or_create(
            subcode=description,
            defaults={'py_file_path': py_file_path, 'pt_file_path': pt_file_path}
        )

        return JsonResponse({'status': 'success', 'message': 'Files uploaded successfully', 'created': created})

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})
