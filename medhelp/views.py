from __future__ import division, print_function

import os

# Keras
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from keras.preprocessing import image
from werkzeug.utils import secure_filename

# Define a flask app

# Model saved with Keras model.save()
from django.contrib import admin
from django.shortcuts import render
import pickle
import numpy as np
from keras.models import load_model
import tensorflow as tf
# Django admin customization
admin.site.site_header = "Login to Cvk007"
admin.site.site_title = "Welcome to Cvk007's Dashboard"
admin.site.index_title = "Welcome folk"
# Model saved with Keras model.save()
MODEL_PATH = 'E:/Chintan_backup/New folder/root/Models/malariaModel.h5'

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Create your views here.
def index(request):
    return render(request, "index.html")


def c_result(request):
    if request.method == 'POST':
        rad = float(request.POST['Radius_mean'])
        tex = float(request.POST['Texture_mean'])
        par = float(request.POST['Perimeter_mean'])
        area = float(request.POST['Area_mean'])
        smooth = float(request.POST['Smoothness_mean'])
        compact = float(request.POST['Compactness_mean'])
        con = float(request.POST['Concavity_mean'])
        concave = float(request.POST['concave points_mean'])
        sym = float(request.POST['symmetry_mean'])
        frac = float(request.POST['fractal_dimension_mean'])
        rad_se = float(request.POST['radius_se'])
        tex_se = float(request.POST['texture_se'])
        par_se = float(request.POST['perimeter_se'])
        area_se = float(request.POST['area_se'])
        smooth_se = float(request.POST['smoothness_se'])
        compact_se = float(request.POST['compactness_se'])
        con_se = float(request.POST['concavity_se'])
        concave_se = float(request.POST['concave points_se'])
        sym_se = float(request.POST['symmetry_se'])
        frac_se = float(request.POST['fractal_dimension_se'])
        rad_worst = float(request.POST['radius_worst'])
        tex_worst = float(request.POST['texture_worst'])
        par_worst = float(request.POST['perimeter_worst'])
        area_worst = float(request.POST['area_worst'])
        smooth_worst = float(request.POST['smoothness_worst'])
        compact_worst = float(request.POST['compactness_worst'])
        con_worst = float(request.POST['concavity_worst'])
        concave_worst = float(request.POST['concave points_worst'])
        sym_worst = float(request.POST['symmetry_worst'])
        frac_worst = float(request.POST['fractal_dimension_worst'])

        data = np.array([[rad, tex, par, area, smooth, compact, con, concave, sym, frac, rad_se, tex_se, par_se,
                          area_se, smooth_se, compact_se, con_se, concave_se, sym_se, frac_se, rad_worst, tex_worst,
                          par_worst, area_worst, smooth_worst, compact_worst, con_worst, concave_worst, sym_worst,
                          frac_worst]])

        my_prediction = pickle.load(open('E:/Chintan_backup/New folder/root/Models/cancer-model.pkl', 'rb')).predict(data)

        return render(request, 'c_result.html', {'prediction': my_prediction})


def cancer(request):
    return render(request, "cancer.html")


def d_result(request):
    if request.method == 'POST':
        preg = int(request.POST['pregnancies'])
        glucose = int(request.POST['glucose'])
        bp = int(request.POST['bloodpressure'])
        st = int(request.POST['skinthickness'])
        insulin = int(request.POST['insulin'])
        bmi = float(request.POST['bmi'])
        dpf = float(request.POST['dpf'])
        age = int(request.POST['age'])

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        d_classifier = pickle.load(open('E:/Chintan_backup/New folder/root/Models/diabetes-model.pkl', 'rb'))
        my_prediction = d_classifier.predict(data)
        return render(request, 'd_result.html', {'my_prediction': my_prediction})


def diabetes(request):
    return render(request, "diabetes.html")


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(130, 130))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')
    images = np.vstack([x])

    preds = model.predict(images, batch_size=16)
    return preds


def h_result(request):
    if request.method == 'POST':

        # Get the file from post request
        f = request.FILES['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads')
        # f.save(file_path)

        fs = FileSystemStorage(location=file_path)  # defaults to   MEDIA_ROOT
        filename = fs.save(f.name, f)
        file_url = fs.url(filename)

        # Make prediction
        preds = model_predict(file_path+file_url, model)

        if preds > 0:
            return HttpResponse("Uninfected")
        else:  # Convert to string
            return HttpResponse("Infected")
    return None

def heart(request):
    return render(request, "heart.html")


def response(request):
    l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
          'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
          'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
          'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
          'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
          'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
          'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
          'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
          'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
          'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
          'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
          'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
          'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
          'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
          'family_history', 'mucoid_sputum',
          'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
          'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
          'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
          'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
          'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
          'yellow_crust_ooze']

    disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
               'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
               ' Migraine', 'Cervical spondylosis',
               'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
               'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
               'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
               'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
               'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
               'Impetigo']

    l2 = []
    for x in range(0, len(l1)):
        l2.append(0)

    psymptoms = [request.POST['ssymp1'], request.POST['ssymp2'], request.POST['ssymp3'],
                 request.POST['ssymp4'], request.POST['ssymp5']]

    for k in range(0, len(l1)):
        # print (k,)
        for z in psymptoms:
            if (z == l1[k]):
                l2[k] = 1

    inputtest = [l2]

    # pickle.dump(clf3, open("DecisionTreeClassifier", 'wb'))
    loaded_model = pickle.load(open("E:/Chintan_backup/New folder/root/Models/DecisionTreeClassifier", 'rb'))
    predict = loaded_model.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if predicted == a:
            h = 'yes'
            break

    if h == 'yes':

        p1 = disease[a]
    else:
        p1 = "Not Found"

    for k in range(0, len(l1)):
        for z in psymptoms:
            if (z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    loaded_model = pickle.load(open("E:/Chintan_backup/New folder/root/Models/RandomForestClassifier", 'rb'))
    predict = loaded_model.predict(inputtest)

    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if (predicted == a):
            h = 'yes'
            break

    if (h == 'yes'):

        p2 = disease[a]
    else:
        p2 = "Not Found"

    for k in range(0, len(l1)):
        for z in psymptoms:
            if (z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    loaded_model = pickle.load(open("E:/Chintan_backup/New folder/root/Models/GaussianNB", 'rb'))
    predict = loaded_model.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if (predicted == a):
            h = 'yes'
            break

    if (h == 'yes'):
        p3 = disease[a]
    else:
        p3 = "Not Found"

    context = {p1, p2, p3}
    return render(request, "index.html", { 'predictions': set(context), 'set': True,
                                          'symptoms': set(psymptoms)})
# data_retrieve
