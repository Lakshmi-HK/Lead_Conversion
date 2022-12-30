import category_encoders as ce
from sklearn.linear_model import LinearRegression
import pandas as pd

def sanity_check(df):
    df.drop_duplicates(inplace=True)
        
    df.drop(columns=['Search', 'Magazine', 'Lead Number', 'Country',
                     'Newspaper Article', 'X Education Forums', 'Newspaper',
                     'Digital Advertisement', 'Through Recommendations','Receive More Updates About Our Courses',
                     'I agree to pay the amount through cheque','Update me on Supply Chain Content',
                     'Get updates on DM Content','Last Activity'],inplace=True)
    return df


def handle_missing_values(df):
    df.dropna(subset=['Lead Source','TotalVisits','Page Views Per Visit'],axis=0,inplace=True)
    df.drop(['Tags','Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score',
         'Asymmetrique Profile Score'] , axis=1,inplace=True )
   
    df.loc[df['Specialization'].isnull() , 'Specialization'] = 'Missing'
    df.loc[df['How did you hear about X Education'].isnull() , 'How did you hear about X Education'] = 'Missing'
    df.loc[df['What is your current occupation'].isnull() , 'What is your current occupation'] = 'Missing'
    df.loc[df['What matters most to you in choosing a course'].isnull() ,'What matters most to you in choosing a course']= 'Missing'
    df.loc[df['Lead Quality'].isnull() , 'Lead Quality'] ='Missing'
    df.loc[df['Lead Profile'].isnull() , 'Lead Profile'] = 'Missing'
    df.loc[df['City'].isnull() , 'City'] = 'Missing'
     
    return df

one_hot_cols = ['Lead Origin' ,  'Do Not Email', 'Do Not Call', 'What is your current occupation', 'What matters most to you in choosing a course',
     'Lead Quality', 'Lead Profile', 'City' , 'A free copy of Mastering The Interview']

def handle_categorical_cols(df):
    ''' This function converts categorical columns into numerical
    Input : dataframe with categorical values
    return: dataframe with transformed categorical values and frequency encoding of a few columns
    
    '''
    df =  hash_encoder(df,'Prospect ID')
    df = one_hot_encoder(df,one_hot_cols)
    df,frequency_dict = frequency_encoder(df,freq_cols)
    return df, frequency_dict


def hash_encoder(df,col):
    # Prospect ID,
    encoder = ce.HashingEncoder(cols= col,n_components= 10)
    hash_encode = encoder.fit_transform(df[col])
    df = pd.concat([df,hash_encode],axis=1)
    df.drop(col,axis=1,inplace=True)
    df.rename(columns={'col_0':"Prospect_ID_0", 'col_1' :"Prospect_ID_1", 'col_2':"Prospect_ID_2", 'col_3':"Prospect_ID_3", 
                'col_4':"Prospect_ID_4",'col_5':"Prospect_ID_5", 'col_6':"Prospect_ID_6", 'col_7':"Prospect_ID_7", 
                'col_8':"Prospect_ID_8", 'col_9':"Prospect_ID_9"},inplace=True)
    return df
    
def one_hot_encoder(df,cols):
    # Lead Origin ,  do not email , do not call, 'What is your current occupation', What matters most to you in choosing a course,
    # lead quality, lead profile, city , 'A free copy of Mastering The Interview
    for i in cols:
        one_hot_encode = pd.get_dummies(data= df[i],prefix = i)
        df = pd.concat([df,one_hot_encode],axis=1)
        df.drop(i,axis=1,inplace=True)
    return df
    
freq_cols = [ 'Lead Source','Country', 'Specialization', 'How did you hear about X Education', 'Last Notable Activity']
def frequency_encoder(df,cols):
    # lead source, country, specialization, How did you hear about X Education, Last Notable Activity
    frequency_dict={}
    for i in cols:
        freq_df = df[i].value_counts(normalize=True)*100
        freq_dict = freq_df.to_dict()
        frequency_dict[i] = freq_dict
        df[i] = df[i].apply(lambda x:freq_dict[x] )
        
    return df, frequency_dict


def sklearn_vif(exogs, data):
    '''
    This function calculates variance inflation function in sklearn way. 
     It is a comparatively faster process.
    '''
    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}

    # form input data for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        X, y = data[not_exog], data[exog]

        # extract r-squared from the fit
        r_squared = LinearRegression().fit(X, y).score(X, y)

        # calculate VIF
        vif = 1/(1 - r_squared)
        vif_dict[exog] = vif

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance

    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    return df_vif

df_vif= sklearn_vif(exogs=X.columns, data=X).sort_values(by='VIF',ascending=False)