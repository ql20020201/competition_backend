from django.apps import AppConfig


class SubmittionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'submittion'

    def ready(self):
        from submittion import attack
        from submittion.models import Submittion, Score 
        s1 = Submittion.objects.all().iterator()

        # 第一个循环遍历表中所有数据
        for s in s1:
            attack_socre = 0
            defend_score = 0
            atk_cnt = 0
            def_cnt = 0
            # 第二个循环也是遍历表里的所有数据
            s2 = Submittion.objects.all().iterator()
            for t in s2:
                # 在这个循环中s1不变，s去攻击t
                try:
                    a, _ = attack('code' + s.subcode, t.subcode)
                    attack_socre += a
                    atk_cnt += 1
                except:
                    pass
                # 在这个循环中，同理t去攻击s
                try:
                    _, d = attack('code' + t.subcode, s.subcode)
                    defend_score += d
                    def_cnt += 1
                except:
                    pass
                print(s.subcode, t.subcode, attack_socre, defend_score)
            try:
                attack_socre /= atk_cnt
                defend_score /= def_cnt
                avgscore = (attack_socre + defend_score) / 2
                print(s.subcode, attack_socre, defend_score, avgscore)
                _, _ = Score.objects.update_or_create(
                    subcode=s.subcode,
                    defaults={'attack': attack_socre, 'defend': defend_score, 'average': avgscore}
                )
            except:
                print('error', s.subcode)
