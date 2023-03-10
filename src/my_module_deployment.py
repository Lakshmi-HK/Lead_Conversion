
import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#from category_encoders import OneHotEncoder, BaseNEncoder
#from category_encoders.binary import BinaryEncoder
le = LabelEncoder()
#b = BinaryEncoder()

def user_report():
    lead_source = st.selectbox('Lead Source', ('Online Chat', 'Organic Search', 'Reference', 'Social Media',
       'Youtube', 'Advertisement'))
    total_visits = st.number_input("Enter Total Visists",0,500)
    total_time = st.number_input("Enter Total Time Spent on Website", 0, 10000)
    page_views = st.number_input("Enter Page Views Per Visit", 0, 100)
    specialization = st.selectbox('Specialization', ('Finance Management', 'Business Administration',
                                                    'Media and Advertising', 'Supply Chain Management',
                                                    'IT Projects Management', 'Travel and Tourism',
                                                    'Human Resource Management', 'Marketing Management',
                                                     'International Business','E-Business', 
                                                     'Operations Management', 'Retail Management',
                                                     'Services Excellence', 'Healthcare Management',
                                                    'Rural and Agribusiness'))

    hear_about = st.selectbox('How did you hear about X Education', ('Online Search', 'Word Of Mouth', 
                                                                     'Other', 'Advertisements',
                                                                     'Email', 'Social Media', 'SMS'))


    last_activity = st.selectbox('Last Notable Activity', ('Modified', 'Email Opened', 'Page Visited on Website',
                                                           'Email Bounced', 'Email Link Clicked', 'Unreachable',
                                                           'Unsubscribed', 'Had a Phone Conversation',
                                                           'Olark Chat Conversation', 'SMS Sent', 'Approached upfront',
                                                           'Resubscribed to emails', 'View in browser link Clicked',
                                                           'Form Submitted on Website', 'Email Received', 
                                                           'Email Marked Spam'))


    user_report_data = {
                             'lead_source': lead_source,
                             'total_visits ': total_visits ,
                             'total_time': total_time,
                             'page_views':  page_views,
                             'specialization': specialization,
                             'hear_about': hear_about,
                             'last_activity': last_activity

                       }

    df = pd.DataFrame(user_report_data, index=[0])
    df['lead_source'] = le.fit_transform(df['lead_source'])
    df['specialization'] = le.fit_transform(df['specialization'])
    df['hear_about'] = le.fit_transform(df['hear_about'])
    df['last_activity'] = le.fit_transform(df['last_activity'])
    #df['last_activity'] = o.fit_transform(np.array(df['last_activity']).reshape(1,-1))
    return df
