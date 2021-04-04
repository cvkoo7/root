from django.db import models

# Create your models here.
class contact(models.Model):

    name = models.TextField()
    email = models.TextField()
    subject = models.TextField()
    patient = models.TextField()

    def __str__(self):
        return self.name