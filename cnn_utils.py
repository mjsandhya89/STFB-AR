from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from imblearn.metrics import specificity_score, sensitivity_score
from sklearn.metrics import accuracy_score

def all_metrics(y_test,y_pred):
    acc=metrics.accuracy_score(y_test, y_pred)
    recall=metrics.recall_score(y_test,y_pred,average='micro') #average : string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
    precision=metrics.precision_score(y_test,y_pred,average='micro')
    f1=metrics.f1_score(y_test, y_pred,average='micro')
    kappa=metrics.cohen_kappa_score(y_test, y_pred)
    # tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    # sensitivity = tp / (tp+fn)
    # specificity = tn / (tn+fp)
    sensitivity=sensitivity_score(y_test, y_pred, average='micro')
    specificity=specificity_score(y_test, y_pred, average='micro')
    return acc,recall,precision,f1,kappa,sensitivity,specificity 
    
def onehot_encode_list(data,data1):
    # define example
    values = array(data)
    values1 = array(data1)

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded1 = label_encoder.transform(values1)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    integer_encoded1 = integer_encoded1.reshape(len(integer_encoded1), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded1 = onehot_encoder.transform(integer_encoded1)
    
    return onehot_encoded,onehot_encoded1

def decode_element(onehot_element):
    inverted = label_encoder.inverse_transform([argmax(onehot_element)])
    return inverted
def decode_list(onehot_list):
    inverted_list=[]
    for onehot_element in onehot_list:
        inverted = label_encoder.inverse_transform([argmax(onehot_element)])
        inverted_list.append(inverted)
    return inverted_list   