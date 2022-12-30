import my_module_deployment
import streamlit as st
import pickle



model = pickle.load(open('../models/model_lightgbm.sav', 'rb'))

st.title('Lead Conversion Status')

user_data = my_module_deployment.user_report()
st.header('Details')
st.write(user_data)

Status = model.predict(user_data)

st.subheader('Conversion Status')

if st.button('Generate Prediction'):
	if Status == 1:
		st.write('YES')
	elif Status == 0:
		st.write('NO')
	else:
		st.write("Please fill inputs ")


