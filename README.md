# Predicting Cardiovascular Disease (CVD)

## Introduction 
Cardiovascular diseases (CVDs) can refer to a number of heart condition that include diseased vessels, structural problems and blood clots. 

####   **Some of the most common conditions are:**
*   Coronary artery disease - Damage or disease in the heart's major blood vessels
*   High blood pressure - A condition in which the force of the blood against the artery walls is too high
*   Cardic arrest - Sudden, unexpected loss of heart function, breathing, and consciousness
*   Congestive heart failure (CHF)- A chronic condition in which the heart doesn't pump blood as well 
*   Arrhymia- Improper beating of the heart, whether irregular, too fast, or too slow.
*   Peripheral artery disease- A circulatory condition in which narrowed blood vessels reduce blood flow to the limbs.
*   Stroke - Damage to the brain from interruption of its blood supply.
*   Congenital heart disease- An abnormality in the heart that develops before birth.



According to the world health organization, "Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year."  More people die annually from CVD's than from any other other causes, representing 31% af all global deaths. 

## Goal
The goal of this study is to create a model to predict the presence or absence of cardiovascular disease (CVD) using the patient's examination results. I would also like to research what factors are primary in predicting if a person has CVD. 

**Other question I would like to explore are:**
* With the increase in age does the chances of heart disease increase.
* Effect of height and weight? We assume the chance of CVD increases with more BMI. 
* Use height and weight to calculate BMI of a patient and see if it has some impact on the target variable.
* Check about how patient drinking and smoking habits would increase the chances of CVD. 
* Effects of increased blood pressure, cholesterol, blood glucose level on CVD. 

## Data 

I obtained the Cardiovascular disease dataset from Kaggle. The origin is not stated. The dataset consists the record of 70,000 patients and 12 features that contains their information. All of the dataset values were collected at the moment of medical examination.

- **The dataset has 3 types of input features:**
- Objective: factual information (patient’s age, weight…)
- Examination: results of medical examination (level of cholesterol, glucose)
- Subjective: information given by the patient (do they smoke, drink…)

| Features | Description | Column name | Datatype
| --- | --- | --| -- |
Age | Objective Feature | age | int (days)
Height | Objective Feature | height | int (cm) |
Weight | Objective Feature | weight | float (kg) |
Gender | Objective Feature | gender | categorical -  code 1: women, 2:men |
Systolic blood pressure | Examination Feature | ap_hi | int |
Diastolic blood pressure | Examination Feature | ap_lo | int |
Cholesterol | Examination Feature | cholesterol | categorical - 1: normal, 2: above normal, 3: well above normal |
Glucose | Examination Feature | gluc | categorical- 1: normal, 2: above normal, 3: well above normal |
Smoking | Subjective Feature | smoke | binary- 0:no, 1:yes|
Alcohol intake | Subjective Feature | alco | binary- 0:no, 1:yes |
Physical activity | Subjective Feature | active | binary- 0:no, 1:yes|
Presence or absence of cardiovascular disease | Target Variable | cardio | binary- 0:no, 1:yes |

The is what the orginal dataset looked like. The top 5 rows.
![alt text]('orginal_df_head5.png')


**The following actions are taken to transfrom the data**

The dataset didn't have any nan values. However, the features of the blood pressure measurements had some pretty high outliers. 

Systolic blood pressure: The top number in blood pressure reading refers to the amount of pressure in your heart when it beats, squeezes and pushes blood through your arteries to the rest of your body. 

Diastolic pressure:  The bottom number in a blood pressure reading refers to the amount of pressure in your arteries when your heart is at rest between beats.

Blood pressure is measured in millimeters of mercury (mm Hg).

[blood pressure](https://www.heart.org/-/media/files/health-topics/high-blood-pressure/hbp-rainbow-chart-english-pdf-ucm_499220.pdf)



According to an article published at the National Library of Medicine - National Center for Biotechnology Information,  the highest blood pressure recorded in an individual was 370/360 (mmHG). I used those numbers to set the limit on my dataset for those variables.

The data had 40 rows that had systolic blood pressure greater than 370, and 953 rows that had diastolic blood pressure greater than 360. I decided to drop those rows - a total of 993. 

## EDA and Feature Engineering 

This section uses data exploratory analysis to investigate features that might be good predictores of whether a person has cvd or not. 

First it will explore the objective features, such as age, BMI and gender. 

**Age**

![age](Count_of_People_Who_Have_not_Have_CVD_for_Each_Age_Group.png)

Age can be a possible predictor for the presence of CVD. As seen from the barplot the count of people who don't have CVD is higher for the age group below 54 years old. As the age increases, after 54 years more people have CVD than those who don't. 


**BMI** 

![bmi](count_bmi.png)

The body mass index is taken by dividing the weight by the height. The barplot shows there are more people who don't have CVD than people who have BMI of less than 27. The trend generally shows that, for those people who have BMI of 27 or larger, there are more people who have CVD than those who don't. While the oppostive is true for when BMI is less than 27.  

**Gender**
![gender](age.png)

The gender doesn't really show much difference. 


Exploration of examination features such as cholestrol and glucose:

![cholestrol](chole_gluc.png)
A cholesterol test is a blood test that measures the amount of each type of cholesterol and certain fats in your blood. The levels are measured in milligrams (mg) of cholesterol per deciliter (dL) of blood. Too much LDL cholesterol in your blood may put you at risk for heart disease and other serious conditions. LDL cholesterol is often called the "bad" cholesterol because it collects in the walls of your blood vessels, raising your chance of heart problems. This is evident in out dataset. As shown in the barplot, the percent of people who have CVD increases with higher cholesterol level. 


Similarly the percent of people who have glucose also increases as glucose level increases. A blood glucose test measures the amount of glucose in your blood.The international standard way of measuring blood glucose levels is in terms of a molar concentration, measured in mmol/L (millimoles per litre; or millimolar, abbreviated mM). High sugar levels slowly erode the ability of cells in your pancreas to make insulin. The organ overcompensates and insulin levels stay too high. Over time, the pancreas is permanently damaged.
High levels of blood sugar can cause changes that lead to a hardening of the blood vessels, what doctors call atherosclerosis, raising your risk of develping CVDs. 


Exploration of subjective features given by the patients, such as smoking, drinking, exercise. 

![smoking](smoking_and_cvd.png)

Majority of the people in the dataset reported that they do not smoke. Not a significant observation can be made between smoking and CVD. Of the people who reported that they do not smoke about half of them have CVD while the other half doesn't. Similary of the people who reported that they do smoke, which is a very small number of people in out dataset, about half of them have cvd. 

![drinking](alcohol_and_cvd.png)

Again, the vast majority of people in the dataset reported that they do not drink alcohol. No strong correlation is observed between the relationship of CVD and alcohol consumption. 

![excerisize](excercise_and_cvd.png)

The majority of people reported that they excercise. Among the people who reported that they are active the presence of CVD is slighly lower by a very small amount. The number of people who have CVD is slighly highter among people who are not active. 

With the initial data EDA, I made the choice to select the following variables to model CVD risk:

- Age
- BMI
- Cholestrol level
- Glucos level
- Systolic blood pressure 
- Diastolic blood pressure 

For Glucose level, and Cholestrol level additional feature engineering is preformed to encode the categorical values. 


### Sampling, Modeling & Comparison

I compared Random Forest Classifier and Gradient Boosting Classifier to see which model has a good predictive power to detect CVD. I also used RandomizedSearchCV  the best parameters for Gradient Boosting classifier using randomed search cv'

GradientBoostingClassifier
























https://pubmed.ncbi.nlm.nih.gov/7741618/#:~:text=The%20highest%20pressure%20recorded%20in,maximal%20lifting%20with%20slow%20exhalation.

https://www.webmd.com/heart-disease/ldl-cholesterol-the-bad-cholesterol#1