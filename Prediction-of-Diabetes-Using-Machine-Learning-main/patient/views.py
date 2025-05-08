from django.shortcuts import render, redirect
from django.contrib.auth.models import User, auth
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from .models import DiabetesData
import threading
import time
import os

# --- Cache variables ---
model = None
scaler = None
model_accuracy = None
loading = True  # Animation flag

# === Animation for race car ===
def racecar_loader():
    car = "🏎️"
    track_length = 30  # Track length
    while loading:
        for pos in range(track_length, -1, -1):  # Start from the rightmost position
            if not loading:
                break
            os.system('cls' if os.name == 'nt' else 'clear')
            print("🏁 Training model... Please wait\n")
            print(" _" * pos + car + "_" * (track_length - pos))  # Move the car from right to left
            time.sleep(0.1)

def initialize_model():
    global model, scaler, model_accuracy, loading
    if model is None or scaler is None:
        loading = True
        t = threading.Thread(target=racecar_loader) # Model training runs only once 
        t.start()

        try:
            df = pd.read_csv("static/dataset/diabetes (1).csv")
            zero_replace_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for col in zero_replace_cols:
                df[col] = df[col].replace(0, df[col].mean())
            df.dropna(inplace=True)

            X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age']]
            y = df['Outcome']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            sm = SMOTE(random_state=42)
            X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }

            grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=0)
            grid.fit(X_train_bal, y_train_bal)
            model = grid.best_estimator_

            y_pred = model.predict(X_test)
            model_accuracy = accuracy_score(y_test, y_pred) * 100
        finally:
            loading = False
            t.join()
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"✅ Model trained well...")

# Initialize model once when the file is loaded
initialize_model()

def home(request):
    return render(request, "index.html")

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return render(request, 'data.html')
        else:
            messages.info(request, 'Invalid credentials')
            return render(request, 'login.html')
    return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        email = request.POST['email']

        if password1 == password2:
            if User.objects.filter(username=username).exists():
                messages.info(request, 'Username already exists')
                return render(request, 'register.html')
            elif User.objects.filter(email=email).exists():
                messages.info(request, 'Email already exists')
                return render(request, 'register.html')
            else:
                User.objects.create_user(
                    username=username,
                    password=password1,
                    email=email,
                    first_name=first_name,
                    last_name=last_name
                )
                return redirect('login')
        else:
            messages.info(request, 'Passwords do not match')
            return render(request, 'register.html')
    return render(request, 'register.html')

@login_required
def predict(request):
    if request.method == 'POST':
        pregnancies = int(request.POST['pregnancies'])
        glucose = int(request.POST['glucose'])
        bloodpressure = int(request.POST['bloodpressure'])
        skinthickness = int(request.POST['skinthickness'])
        insulin = int(request.POST['insulin'])
        bmi = float(request.POST['bmi'])
        diabetespedigreefunction = float(request.POST['diabetespedigreefunction'])
        age = int(request.POST['age'])

        input_data = [[pregnancies, glucose, bloodpressure, skinthickness,
                       insulin, bmi, diabetespedigreefunction, age]]
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        DiabetesData.objects.create(
            Pregnancies=pregnancies,
            Glucose=glucose,
            BloodPressure=bloodpressure,
            SkinThickness=skinthickness,
            Insulin=insulin,
            BMI=bmi,
            DiabetesPedigreeFunction=diabetespedigreefunction,
            Age=age
        )


        if probability >= 0.85:
            result_text = "⚠️ Patient is affected by diabetes."
            remedy = [
                "• Start insulin or oral diabetes medication as prescribed.",
                "• Strictly avoid high-sugar and high-carb foods.",
                "• Check blood sugar levels multiple times a day.",
                "• Consult an endocrinologist regularly.",
                "• Monitor carbohydrate intake and consider using a carbohydrate counting method.",
                "• Stay hydrated by drinking plenty of water throughout the day.",
                "• Educate yourself about diabetes management and treatment options.",
                "• Keep a diabetes management journal to track food, activity, and blood sugar levels.",
                "• Consider joining a diabetes support group for emotional and practical support.",
                "• Work with a registered dietitian to create a personalized meal plan."
            ]
        elif 0.65 <= probability < 0.85:
            result_text = "⚠️ Patient is affected by diabetes but at an earlier stage."
            remedy = [
                "• Adopt a low glycemic index diet immediately.",
                "• Begin a daily exercise routine like walking or swimming.",
                "• Use natural aids like fenugreek and cinnamon in meals.",
                "• Have regular health checkups every 3-6 months.",
                "• Monitor blood sugar levels periodically to understand trends.",
                "• Increase physical activity by incorporating more movement into daily routines.",
                "• Limit processed foods and focus on whole, unprocessed foods.",
                "• Manage stress through relaxation techniques such as yoga or meditation.",
                "• Educate yourself about the risk factors and symptoms of diabetes.",
                "• Consider working with a healthcare professional to develop a prevention plan."
            ]
        elif 0.35 <= probability < 0.65:
            result_text = "✅ Patient is not affected by diabetes but at risk in the future."
            remedy = [
                "• Avoid excessive sugar intake and processed foods.",
                "• Include fiber-rich foods like oats, legumes, fruits, and leafy greens.",
                "• Choose whole grains over refined grains.",
                "• Incorporate healthy fats from avocados, nuts, and olive oil.",
                "• Stay physically active with at least 150 minutes of moderate aerobic activity weekly.",
                "• Include strength training exercises twice a week.",
                "• Consider walking, cycling, swimming, or dancing to make exercise enjoyable.",
                "• Limit screen time and sedentary behavior.",
                "• Improve sleep habits (7-9 hours nightly).",
                "• Manage stress with yoga, meditation, or deep breathing.",
                "• Schedule regular check-ups with a healthcare provider.",
                "• Consider periodic blood tests to assess glucose and other markers.",
                "• Keep a food diary to track eating habits.",
                "• Educate yourself on diabetes prevention strategies.",
                "• Join community health programs for healthy living.",
                "• Consult a registered dietitian for personalized advice."
            ]
        else:
            result_text = "✅ Patient is not affected by diabetes."
            remedy = [
                "• Continue balanced diet and regular hydration.",
                "• Get 7–8 hours of sleep daily.",
                "• Stay active — walking, jogging, or yoga helps a lot.",
                "• Get a diabetes check once a year for early detection.",
                "• Incorporate more fruits and vegetables into meals.",
                "• Limit processed foods and sugary snacks.",
                "• Maintain a healthy weight through diet and exercise.",
                "• Stay informed about diabetes prevention strategies.",
                "• Manage stress through mindfulness practices.",
                "• Avoid smoking and limit alcohol consumption.",
                "• Consider regular health screenings for blood pressure and cholesterol.",
                "• Join healthy living community activities.",
                "• Keep a food diary for diet tracking.",
                "• Stay socially active for mental well-being.",
                "• Educate yourself on healthy cooking methods.",
                "• Set realistic health goals and track progress."
            ]

        return render(request, 'predict.html', {
            "data": result_text,
            "pregnancies": pregnancies,
            "glucose": glucose,
            "bloodpressure": bloodpressure,
            "skinthickness": skinthickness,
            "insulin": insulin,
            "bmi": bmi,
            "diabetespedigreefunction": diabetespedigreefunction,
            "age": age,
            "remedy": remedy,
            "model_accuracy": round(model_accuracy, 2),
            "input_accuracy": round(probability * 100, 2)
        })

    return render(request, 'predict.html')
