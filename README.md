# Wine_Quality_Predication_App

In this project, we aim to delve into the wine quality analysis to understand the factors that contribute to the wine qualtiy. By conducting thorough exploratory data analysis, we aim to uncover patterns, correlations, and insights that find out what features might impact the wine qualtiy. We will also utilize machine learning techniques to predict wine quality. At the end, we created a wine quality prediction app using Streamlit with the best performance machine learning model(Random Forest).

There are two part of this project: one is the notebook for wine quality analysis and prediction using machine learning techniques, the other is the application part for wine quality prediction app.

# The wine quality prediction App:
You can check any wine quality with modifying the wine features parameters on the left side. The quality class starting from 3 to 9 while 3 is the lowest and 9 is the highest. Under the prection section, you can see what wine quality class.
![wine_quality_pred_app](https://github.com/lynsitu/Wine_Quality_PredicationApp/assets/61662998/bdabbd93-38a7-438f-9165-31ffca958148)

# How to run the app on your local machine:
Step 1: git clone {https link for clone} to copy all the folders.

Step 2: Create the virtual environment on your machine
```
$ virtualenv venv -p Python3
```

Step 3: Activate the virtual environment
```
$ source venv/bin/activate
```

Step 4: Install the dependencies in the requirements.txt file
```
pip install -r requirements.txt
```

Step 5: Now you can run the app.py
```
streamlit run app.py
```
