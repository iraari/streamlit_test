import shap
from streamlit_shap import st_shap
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings


df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")#("train_ajEneEa.csv")
keep_columns = ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'HeartDiseaseorAttack', 'Veggies',
                'HvyAlcoholConsump', 'GenHlth', 'DiffWalk', 'Sex', 'Age', 'Income']
df = df[keep_columns]

df = df.rename(columns={"HighBP": "High_blood_pressure", "HighChol": "High_cholesterol", 
                        "HeartDiseaseorAttack": "Heart_disease_or_attack", 
                        "HvyAlcoholConsump": "Heavy_alcohol_consumption", 
                        "GenHlth": "General_health", #"PhysHlth": "physical_health",
                        "DiffWalk": "Difficulty_walking"})

df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

# split the training dataset into X and y
X_train = df_train.drop(['Diabetes_binary'], axis=1)
y_train = df_train['Diabetes_binary']

X_test = df_test.drop(['Diabetes_binary'], axis=1)
y_test = df_test['Diabetes_binary']

nums = ['High_blood_pressure', 'High_cholesterol', 'BMI', 'Heart_disease_or_attack', 'Veggies',
        'Heavy_alcohol_consumption', 'General_health', #'physical_health', 
        'Difficulty_walking', 'Sex', 'Age', 'Income']
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), nums)])
model_LR = Pipeline([('preprocessor', preprocessor),
                     ('classifier', LogisticRegression(solver='liblinear', C=1.0, max_iter=10000))])
model_LR.fit(X_train, y_train)



#st.write("Hello world")

"""
# Diabetes Health Indicators
Hello!

This is an application that can serve as a self-regulation and check whether a user is likely to develop diabetes. 

It is based on a machine learning model that makes predictions based on patterns in persons's health data.

### Introduction

Here’s how it works:

- Learning from Examples: First, the model is trained using a large set of health records collected previously. Each record has details about a person’s health and a label indicating whether they have diabetes (1) or not (0). 

- Finding Patterns: As the model processes more examples, it notices which factors—like high blood pressure or certain age ranges—are often seen in people with diabetes. It also learns which patterns are typical in people who don’t have diabetes.

- Making Predictions: Once it’s trained, the model can then be used with new health records. When given the details of a new person, it checks the patterns in that person’s data and compares them with the patterns it has learned. If the data aligns closely with patterns seen in people with diabetes, the model predicts "1" (meaning the person likely has diabetes). If it doesn’t match those patterns, it predicts "0" (meaning the person likely does not have diabetes).

- Confidence Level: The model in this study gives a confidence level, like saying there’s an 80% chance the person has diabetes. This helps understand how strongly the model "believes" in its prediction.

In short, a machine learning model for diabetes prediction works by learning from many examples to find patterns, then uses those patterns to make educated guesses about new cases. This doesn’t replace a doctor's expertise but can serve as a helpful tool to support decision-making.



### Would you be interested to know why the model came to a certain result? 

There are different tools that help people understand why a machine learning model made a particular prediction. This study tests one of them called "SHAP".

Think of SHAP as a way to "look inside" the model to see which factors were important in making its decision.
Since machine learning models are often complex and don't explain themselves, SHAP steps in to solve this problem by giving each feature (such as blood pressure or age) a score, called a "SHAP value." 
This score tells us how much that specific factor contributed to the final prediction.


### Please proceed as following:
- fill in the data to send it to the model
- after you get the result, view the explanations 
- share your thoughts :)

Important note: this is not a test of your knowledge here. This is a test of the interpretability of the method I'm researching. 

Your feedback and thoughts on the format of the explanations is very valuable and I encourage you to say whatever you think. 
That way we can determine the shortcomings and strengths of the method and find potential areas for improvement. 

Thanks for agreeing to participate in this study!

"""


HighBP = st.selectbox(
    'Do you have high blood pressure? 0 = no high BP, 1 = high',
     ("no", "yes"))
#'You selected: ', HighBP
HighBP_dict =  {"no": 0, "yes": 1}
#st.write(f"key for value = {HighBP_dict.get(HighBP)}")

HighChol = st.selectbox(
    'Is yout cholesterol high? 0 = no high cholesterol, 1 = high cholesterol',
     ("no", "yes"))
#'You selected: ', HighChol
HighChol_dict =  {"no": 0, "yes": 1}
#st.write(f"key for value = {HighChol_dict.get(HighChol)}")

BMI = st.slider('What is your Body Mass Index?')  # 👈 this is a widget
#st.write('your BMI ', BMI)

HeartDiseaseorAttack = st.selectbox(
    'Do you have coronary heart disease (CHD) or myocardial infarction (MI)? 0 = no, 1 = yes',
     ("no", "yes"))
#'You are ', HeartDiseaseorAttack
HeartDiseaseorAttack_dict =  {"no": 0, "yes": 1}
#st.write(f"key for value = {HeartDiseaseorAttack_dict.get(HeartDiseaseorAttack)}")

Veggies = st.selectbox(
    'Do you consume Vegetables 1 or more times per day 0 = no 1 = yes',
     ("no", "yes"))
'You are ', Veggies
Veggies_dict =  {"no": 0, "yes": 1}
st.write(f"key for value = {Veggies_dict.get(Veggies)}")

HvyAlcoholConsump = st.selectbox(
    'Do you have more than 14 drinks per week (adult men) or more than 7 drinks per week (adult women)? 0 = no, 1 = yes',
     ("no", "yes"))
#'You are ', HvyAlcoholConsump
HvyAlcoholConsump_dict =  {"no": 0, "yes": 1}
#st.write(f"key for value = {HvyAlcoholConsump_dict.get(HvyAlcoholConsump)}")

GenHlth = st.radio(
        'Would you say that in general your health is:',
        ("excellent", "very good", "good", "fair", "poor"))
#st.write(f"Your GenHlth {GenHlth}")
GenHlth_dict =  {"excellent": 1, "very good": 2, "good": 3, "fair": 4, "poor": 5}
#st.write(f"key for value = {GenHlth_dict.get(GenHlth)}")

#PhysHlth = st.slider('Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days', min_value=0, max_value=30, step=1)  # 👈 this is a widget
#st.write('your PhysHlth ', PhysHlth)

DiffWalk = st.selectbox(
    'Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes',
     ("no", "yes"))
#'You selected: ', DiffWalk
DiffWalk_dict =  {"no": 0, "yes": 1}
#st.write(f"key for value = {DiffWalk_dict.get(DiffWalk)}")

Sex = st.radio(
        'What is your sex? 0 = female, 1 = male',
        ("female", "male"))
#st.write(f"You are {Sex}")
Sex_dict =  {"female": 0, "male": 1}
#st.write(f"key for value = {Sex_dict.get(Sex)}")

Age = st.radio(
        'Please select your age range:',
        ("18 to 24 years (1)", "25 to 29 years (2)", "30 to 34 years (3)", "35 to 39 years (4)", "40 to 44 years (5)", 
         "45 to 49 years (6)", "50 to 54 years (7)", "55 to 59 years (8)", "60 to 64 years (9)", "65 to 69 years (10)", 
         "70 to 74 years (11)", "75 to 79 years (12)", "80 years or older (13)"))
#st.write(f"You are {Age}")

Age_dict = {"18 to 24 years (1)": 1, "25 to 29 years (2)": 2, "30 to 34 years (3)": 3, "35 to 39 years (4)": 4, 
            "40 to 44 years (5)": 5, "45 to 49 years (6)": 6, "50 to 54 years (7)": 7, "55 to 59 years (8)":8, 
            "60 to 64 years (9)": 9, "65 to 69 years (10)": 10, "70 to 74 years (11)": 11, "75 to 79 years (12)": 12, 
            "80 years or older (13)": 13}
#st.write(f"key for value = {Age_dict.get(Age)}")

Income = st.radio(
        'Please select your income range:',
        ("Less than \$10,000 (1)", "\$10,000 to less than \$15,000 (2)", "\$15,000 to less than \$20,000 (3)", 
         "\$20,000 to less than \$25,000 (4)", "\$25,000 to less than \$35,000 (5)", 
         "\$35,000 to less than \$50,000 (6)", "\$50,000 to less than \$75,000 (7)", "\$75,000 or more (8)"))
#st.write(f"You make {Income}")

Income_dict = {"Less than \$10,000 (1)": 1, "\$10,000 to less than \$15,000 (2)": 2, "\$15,000 to less than \$20,000 (3)": 3, 
               "\$20,000 to less than \$25,000 (4)": 4, "\$25,000 to less than \$35,000 (5)": 5, 
               "\$35,000 to less than \$50,000 (6)": 6, "\$50,000 to less than \$75,000 (7)": 7, 
               "\$75,000 or more (8)": 8}

#st.write(f"key for value = {Income_dict.get(Income)}")

user_show = {'High_blood_pressure': HighBP,
 'High_cholesterol': HighChol,
 'BMI': BMI,
 'Heart_disease_or_attack': HeartDiseaseorAttack,
 'Veggies': Veggies,
 'Heavy_alcohol_consumption': HvyAlcoholConsump,
 'General_health': GenHlth,
 #'physical_health': "not good for " + str(PhysHlth) + " days",
 'Difficulty_walking': DiffWalk,
 'Sex': Sex,
 'Age': Age,
 'Income': Income}
       
user = {'High_blood_pressure': float(HighBP_dict.get(HighBP)),
 'High_cholesterol': float(HighChol_dict.get(HighChol)),
 'BMI': BMI,
 'Heart_disease_or_attack': float(HeartDiseaseorAttack_dict.get(HeartDiseaseorAttack)),
 'Veggies': float(Veggies_dict.get(Veggies)),
 'Heavy_alcohol_consumption': float(HvyAlcoholConsump_dict.get(HvyAlcoholConsump)),
 'General_health': float(GenHlth_dict.get(GenHlth)),
 #'physical_health': PhysHlth,
 'Difficulty_walking': float(DiffWalk_dict.get(DiffWalk)),
 'Sex': float(Sex_dict.get(Sex)),
 'Age': float(Age_dict.get(Age)),
 'Income': float(Income_dict.get(Income))}


df_user_show = pd.DataFrame([user_show])
df_user_show_T = df_user_show.T.rename(columns={0: "User"})
st.dataframe(df_user_show_T, height=420, width=450) 


df_small = pd.DataFrame([user])
#st.dataframe(df_small) 

#st.write(user)

if st.button('Predict'):
    result = model_LR.predict_proba(df_small)[0, 1]
    st.success("There's a {} \% chance the person has diabetes".format(round(result*100, 2)))


if st.button('Explain!'):

    X_sub = shap.sample(X_train, 1000)
    ex = shap.Explainer(model_LR.predict_proba, X_sub)
    #shap_values = ex(X_test.iloc[0:100])
    #shap_values = ex(X_test.iloc[[-1]])
    shap_values = ex(df_small)

    shap_values.display_data = df_user_show.values

    st.write("### Local explanations")
    class_index = 1
    data_index = 0
    st_shap(shap.plots.waterfall(shap_values[data_index,:,class_index]), height=300)

    #st_shap(shap.plots.waterfall(shap_values[0]), height=600)

    #class_index = 0
    #st_shap(shap.plots.waterfall(shap_values[data_index,:,class_index]), height=300)

    class_index=1
    st_shap(shap.plots.bar(shap_values[data_index,:,class_index]), height=300)

    shap.initjs()
    st_shap(shap.plots.force(shap_values[data_index,:,class_index]))


    st.write("### Global explanations")

    #global
    shap_values_global = ex(X_test.iloc[0:1000])
    st_shap(shap.plots.beeswarm(shap_values_global[:,:,class_index]))


    # st_shap(shap.plots.heatmap(shap_values_global[:,:,class_index]), height=300, width=300)
    # [Error] No plot to display. Unable to understand input.

    #df_display = pd.read_csv("df_display.csv")
    #shap_values_global.display_data = df_display.drop(['Diabetes_binary'], axis=1).iloc[X_test.iloc[0:20].index].values

    shap.initjs()
    st_shap(shap.plots.force(shap_values_global[0:20:,:,1]), height=300)
    
    # train an XGBoost model
    import xgboost
    #X, y = shap.datasets.california()
    model = xgboost.XGBRegressor().fit(X_train, y_train)

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model)
    shap_values_xgboost = explainer(X_train)

    st_shap(shap.plots.scatter(shap_values_xgboost[:, "BMI"], color=shap_values_xgboost))