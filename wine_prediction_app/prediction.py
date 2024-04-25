import joblib
# from prediction import predict

def predict(data):
    clf = joblib.load('model.pkl')
    return clf.predict(data)
