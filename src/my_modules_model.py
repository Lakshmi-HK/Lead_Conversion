import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_recall_curve , roc_curve
import pickle
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.preprocessing import PowerTransformer


path = "../data/Leads.csv"

def load_df(path):
    df = pd.read_csv(path)
    print('Data uploaded sucessfully')
    return df

def sanity_check(df):
    df.drop_duplicates(inplace=True)
        
    df.drop(columns=['A free copy of Mastering The Interview', 'Asymmetrique Activity Index',
                    'Asymmetrique Activity Score', 'Asymmetrique Profile Index',
                    'Asymmetrique Profile Score','City','Country','Digital Advertisement',
                    'Do Not Call','Do Not Email','Get updates on DM Content',
                    'I agree to pay the amount through cheque','Last Activity','Lead Number',
                    'Lead Origin','Lead Profile','Lead Quality','Magazine','Newspaper',
                    'Newspaper Article','Prospect ID','Receive More Updates About Our Courses',
                    'Search','Tags','Through Recommendations','Update me on Supply Chain Content',
                    'What is your current occupation','What matters most to you in choosing a course',
                    'X Education Forums'],inplace=True)
    return df


def handle_missing_values(df):
    df.dropna(subset=['Lead Source','TotalVisits','Page Views Per Visit'],axis=0,inplace=True)
    
    df.loc[df['Specialization'].isnull() , 'Specialization'] = 'Missing'
    df.loc[df['How did you hear about X Education'].isnull() , 'How did you hear about X Education'] = 'Missing'
   
    return df


def handle_categorical_cols(df):
    ''' This function converts categorical columns into numerical
    Input : dataframe with categorical values
    return: dataframe with transformed categorical values and frequency encoding of a few columns
    
    '''
    dictt = {  
              'Google': 'Organic Search','google': 'Organic Search','Welingak Website' : 'Organic Search',
              'Referral Sites' : 'Reference','Facebook': 'Social Media','blog': 'Youtube',
              'youtubechannel': 'Youtube','Missing': 'Google','welearnblog_Home' : 'Organic Search' , 
              'Pay per Click Ads': 'Advertisement','Click2call': 'Advertisement',
              'Press_Release': 'Advertisement','NC_EDM':'Advertisement',
              'Olark Chat': 'Online Chat','Live Chat': 'Online Chat','testone' :'Organic Search',
              'Direct Traffic' : 'Organic Search','bing' : 'Reference',
              'WeLearn':'Organic Search'
            }
    df['Lead Source'] = df['Lead Source'].replace(dictt) 
    df.loc[df['Specialization'] == 'Select', 'Specialization'] = 'Missing'
    df.loc[df['Specialization'] == 'Missing', 'Specialization'] = 'Finance Management'

    spec_dict= { 
                    'E-COMMERCE' : 'E-Business','Hospitality Management': 'Healthcare Management',
                    'Banking, Investment And Insurance': 'Finance Management' 
               }
    df['Specialization'] = df['Specialization'].replace(spec_dict)

    df.loc[df['How did you hear about X Education'].isnull(), 'How did you hear about X Education'] = 'Missing'
    df.loc[df['How did you hear about X Education'] == 'Select', 'How did you hear about X Education'] = 'Missing'
    hear_dict = {
                    'Missing' : 'Online Search', 'Student of SomeSchool': 'Word Of Mouth',
                    'Multiple Sources': 'Other'
                }
    df['How did you hear about X Education'] = df['How did you hear about X Education'].replace(hear_dict)

    df,frequency_dict = frequency_encoder(df,freq_cols)
    return df, frequency_dict


freq_cols = [ 'Lead Source','Specialization', 'How did you hear about X Education', 'Last Notable Activity']
def frequency_encoder(df,cols):
    # lead source, country, specialization, How did you hear about X Education, Last Notable Activity
    frequency_dict={}
    for i in cols:
        freq_df = df[i].value_counts(normalize=True)*100
        freq_dict = freq_df.to_dict()
        frequency_dict[i] = freq_dict
        df[i] = df[i].apply(lambda x:freq_dict[x] )
        
    return df, frequency_dict


def outlier_handle(X):
    X_transform = PowerTransformer().fit_transform(X)
    X_transformed = pd.DataFrame(X_transform,columns=X.columns)
    return X_transformed


def Train_model(x_train, y_train, model):
    model.fit(x_train, y_train)
    return print('model trained sucessfully')


def evalaute_model(y_test, predictions):
    print(f"Recall: {round(recall_score(y_test, predictions) * 100, 4)}")
    print(f"Precision: {round(precision_score(y_test, predictions) * 100, 4)}")
    print(f"F1-Score: {round(f1_score(y_test, predictions) * 100, 4)}")
    print(f"Accuracy score: {round(accuracy_score(y_test, predictions) * 100, 4)}")
    print(f"AUC Score: {round(roc_auc_score(y_test, predictions) * 100, 4)}")


def save_model(model):
    pickle.dump(model, open('../models/model_lightgbm.sav', 'wb'))
    print('model saved')


def confusion_matrix(y_train_test, y_train_test_pred):
    conf = ConfusionMatrixDisplay.from_predictions(y_train_test, y_train_test_pred)
    return conf


def plot_precision_recall_vs_threshold(x_train_test, y_train_test, model):
    y_scores = model.predict_proba(x_train_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train_test, y_scores)
    plt.figure(figsize=(12,6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


#def plot_roc_curve(fpr, tpr, label=None):
def plot_roc_curve(x_test , y_test, predictions,model):
    y_scores = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    plt.figure(figsize=(12,6))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
   