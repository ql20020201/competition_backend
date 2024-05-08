from django.db import models


# 定义了数据库的结构 Defined the structure of the database Create your models here.
class Submittion(models.Model):
    subcode = models.CharField(max_length=255, primary_key=True)
    py_file_path = models.CharField(max_length=500)
    pt_file_path = models.CharField(max_length=500)

class Score(models.Model):
    subcode = models.CharField(max_length=255, primary_key=True)
    attack = models.FloatField()
    defend = models.FloatField()
    average = models.FloatField()

class User(models.Model):
    subcode = models.CharField(max_length=255, primary_key=True)
    usrname = models.CharField(max_length=255, null=False)
    student_no = models.CharField(max_length=255, null=False)