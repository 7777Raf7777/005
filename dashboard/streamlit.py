import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import plotly.io as pio
pio.templates["jedha"] = go.layout.Template(layout_colorway=["#4B9AC7", "#4BE8E0", "#9DD4F3", "#97FBF6", "#2A7FAF", "#23B1AB", "#0E3449", "#015955"])
pio.templates.default = "jedha"
import numpy as np


### Config
st.set_page_config(
    page_title="Getaround",
    page_icon="🚘 ",
    layout="wide"
)

st.title("🚘 Getaround")

st.markdown("""
Welcome ! This is a dashboard that will help the Getaround's product Management team to take the best decisions 
about the minimum delay between two car rentals.
""")


#@st.cache_data(allow_output_mutation=True)
def load_data(nrows):
    data = pd.read_excel('get_around_delay_analysis.xlsx')
   
    return data


data_load_state = st.text('Loading data...')
data = load_data(1000)
data_load_state.text("") # change text from "Loading data..." to "" once the the load_data function has run

## Run the below code if the check is checked ✅
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)  


st.header("Commençons par quelques statistiques sur la location de voitures")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Pourcentage de retards de location par type de check-in**")
    data['late_checkout'] = data['delay_at_checkout_in_minutes'].apply(lambda x: 'en retard' if x > 0 else 'à temps')
    checkout = st.selectbox("Sélectionner un type de check-in", ['tous', 'mobile', 'connect'], key=1)
    late_df = data if checkout == 'tous' else data[data["checkin_type"]==checkout]
    delay_percentage = round(len(late_df[late_df['late_checkout']=='en retard']) / late_df['late_checkout'].count() * 100)
    st.markdown(f'{delay_percentage}% des locations sont en retard avec le type de check-in {checkout}.')
    fig = go.Figure(data=[go.Pie(labels=late_df['late_checkout'])])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Pourcentage des locations annulées par type de check-in**")
    checkout = st.selectbox("Sélectionner un type de check-in", ['tous', 'mobile', 'connect'], key=2)
    canceled_df = data if checkout == 'tous' else data[data["checkin_type"]==checkout]
    canceled_percentage = round(canceled_df['state'].value_counts()[1] / len(canceled_df) * 100)
    st.markdown(f'{canceled_percentage}% des locations sont annulées avec le type de check-in {checkout}.')
    fig = go.Figure(data=[go.Pie(labels=canceled_df['state'])])
    st.plotly_chart(fig, use_container_width=True)


  

data["delay"]=data["delay_at_checkout_in_minutes"].apply(lambda x : "After time/Late" if x>0 else "In advance/On time")
    
    
col1, col2 = st.columns(2)

with col1:
    
        st.subheader("Type de check-in et statut de la location")
        fig = px.histogram(data, x="state",
                   color="checkin_type",
                   barmode="group",
                   width=800,
                   height=600,
                   histnorm="percent",
                   text_auto=True)
        fig.update_traces(textposition="outside", textfont_size=15)
        fig.update_layout(margin=dict(l=50, r=50, b=50, t=50, pad=4),
                  yaxis={"visible": False},
                  xaxis={"visible": True})
        fig.update_xaxes(tickfont_size=15)
        st.plotly_chart(fig)