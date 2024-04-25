#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib

#reading csv file
white_df = pd.read_csv('winequality-white.csv', sep=';')
red_df = pd.read_csv('winequality-red.csv', sep=';')
white_df['type'] = 'white'
red_df['type'] = 'red'
data  = pd.concat([red_df, white_df], axis=0)

# Split into X (features) and y (target)
X = data.drop(columns=['type', 'quality'])
y = data['quality']

oversample = SMOTE(k_neighbors=4)
# transform the datatset
X_B, y_B = oversample.fit_resample(X, y)

def normalize(x_train, x_test):
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)
  return x_train_scaled, x_test_scaled

def classify(model, X_B, y_B):
  x_train, x_test, y_train, y_test = train_test_split(X_B, y_B, test_size=0.2, random_state=42)
  normalize(x_train, x_test)
  model.fit(x_train, y_train)
  y_predict = model.predict(x_test)
  print(classification_report(y_test, y_predict)) 

model_rf = RandomForestClassifier(n_estimators=500, random_state=42)
classify(model_rf, X_B, y_B)

# Save the model as a pickle in a file 
joblib.dump(model_rf, 'model.pkl') 
  