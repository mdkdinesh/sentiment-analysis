from django.db import models


# Create your models here.

class Pos_Freq(models.Model):
    word = models.CharField(max_length=100, null=True)
    freq = models.IntegerField(null=True)


class Table_data(models.Model):
    date = models.CharField(max_length=20)
    city = models.CharField(max_length=20)
    star = models.CharField(max_length=2)
    review = models.CharField(max_length=100)
    sentiment = models.CharField(max_length=20, null=True)
