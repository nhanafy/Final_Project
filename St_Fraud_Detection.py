import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from st_pages import Page, show_pages, add_page_title

#import xgboost as xgb
from streamlit_option_menu import option_menu

#---------------------------- Hide Streamlit Styles ------------------------------

hide_st_style = '''
                <style>
                #MainMenu {visibility : hidden;}
                footer {visibility : hidden;}
                header (visibility : hidden;)
                </style>
                '''

st.markdown(hide_st_style , unsafe_allow_html= True)

# --------------------------2. horizontal menu -----------------------------
h_menu = option_menu(None, ["Home", "Upload"],#'Settings'], 
    icons=['house', 'cloud-upload'], #'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        #"Container": {"background-color": "#fafafa","color":"black"},
        "icon": { "font-size": "25px"},
        "nav-link-selected": {"background-color": "#00B1C3"},
    }
)
#--------------------------------Header Image ---------------------------------


st.write("# Ethereum Fraud Detection")

img = Image.open('Images/ethereum.jpg')
st.image(img,
        use_column_width= True)




if h_menu == "Home":
    st.markdown("""
                ### Introduction
    Fraudulent activities, such as money laundering, bribery, and phishing, emerge as the primary threat to trade security.

    ### What is Ethereum?
    Ethereum is a decentralized blockchain platform that establishes a peer-to-peer network that securely executes and verifies application code, called smart contracts. Smart contracts allow participants to transact with each other without a trusted central authority, founded by Vitalik Buterin and Gavin Wood in 2015, today Ethereum’s market capitalization represents more than 17% of the $1.2 trillion global crypto market.
    Ether (ETH), the native cryptocurrency of the Ethereum network, is the second most popular digital token after bitcoin (BTC)

    ### What is ERC20?
    Ethereum Request for Comment 20 (ERC-20) is the implemented standard for fungible tokens created using the Ethereum blockchain. ERC-20 guides the creation of new tokens on the Ethereum blockchain so that they are interchangeable with other smart contract tokens.
    ERC20 is the standard protocol for creating Ethereum-based tokens, which can be utilized and deployed in the Ethereum network.
    """)  
# 1. as sidebar menu
# with st.sidebar:
#     selected = option_menu('',options=["Home", 'Settings'], 
#         icons=['house', 'gear'], menu_icon="cast", default_index=1)
#     selected

# 2. horizontal menu

# 3. CSS style definitions
# selected3 = option_menu(None, ["Home", "Upload",  "Tasks", 'Settings'], 
#     icons=['house', 'cloud-upload', "list-task", 'gear'], 
#     menu_icon="cast", default_index=0, orientation="horizontal",
#     styles={
#         "container": {"padding": "0!important", "background-color": "#fafafa"},
#         "icon": {"color": "orange", "font-size": "25px"}, 
#         "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
#         "nav-link-selected": {"background-color": "green"},
#     }
# )

class preproc_C:

    def __init__(self,dframe) -> None:
        for i in dframe.columns:
            dframe.rename(columns=str.strip,inplace = True)
    
        droplst = ['Index',
                'Address',
                'ERC20 most sent token type',
                'ERC20_most_rec_token_type',
                'ERC20 avg time between sent tnx',
                'ERC20 avg time between rec tnx',
                'ERC20 avg time between rec 2 tnx',
                'ERC20 avg time between contract tnx',
                'ERC20 min val sent contract',
                'ERC20 max val sent contract',
                'ERC20 avg val sent contract',
                'total transactions (including tnx to create contract',
                'total ether sent contracts', 
                'max val sent to contract',
                'ERC20 avg val rec',
                'ERC20 avg val rec',
                'ERC20 max val rec', 
                'ERC20 min val rec', 
                'ERC20 uniq rec contract addr', 
                'max val sent', 
                'ERC20 avg val sent',
                'ERC20 min val sent', 
                'ERC20 max val sent', 
                'Total ERC20 tnxs', 
                'avg value sent to contract', 
                'Unique Sent To Addresses',
                'Unique Received From Addresses', 
                'total ether received', 
                'ERC20 uniq sent token name', 
                'min value received', 
                'min val sent', 
                'ERC20 uniq rec addr',
                'ERC20 uniq sent addr.1',
                'min value sent to contract']
        dframe.drop(droplst, axis=1, inplace=True)
        
    def fill_median(df):
        medians = {"Avg min between sent tnx":    1.752500e+01,
        "Avg min between received tnx": 5.020400e+02,
        "Time Diff between first and last (Mins)":4.608212e+04,
        "Sent tnx" :    3.000000e+00,
        "Received Tnx" :4.000000e+00,
        "Number of Created Contracts":0.000000e+00,
        "max value received" : 6.000000e+00,
        "avg val received": 1.720991e+00,
        "avg val sent"  : 1.557448e+00,
        "total Ether sent"   : 1.265276e+01,
        "total ether balance" :1.722000e-03,
        "ERC20 total Ether received": 9.110000e-16,
        "ERC20 total ether sent": 0.000000e+00,
        "ERC20 total Ether sent contract":  0.000000e+00,
        "ERC20 uniq sent addr":  0.000000e+00,
        "ERC20 uniq rec token name":1.000000e+00}

        for i in medians.keys():
            df[i].fillna(medians[i], inplace=True)
            
        return df
    

if h_menu == "Upload":
    spectra = st.file_uploader("upload file", type={"csv"})
    if spectra is not None:
        spectra_df = pd.read_csv(spectra, index_col=0)
        addresses_df = spectra_df['Address']

        # st.write(spectra_df.shape)
        proc= pickle.load(open('preproc_C.pkl', 'rb')) 
        preproc = preproc_C(spectra_df)
        X1 = spectra_df.iloc[:, 1:]
        X1_MMs = preproc.fill_median(X1)
        MM_Scaler = pickle.load(open('scaler.pkl','rb'))
        X1_MMs= MM_Scaler.transform(X1_MMs)
        XGBm = pickle.load(open('model.pkl', 'rb'))
        preds = XGBm.predict(X1_MMs)
        pred = pd.Series(preds)
        pred = pd.DataFrame(pred , columns=['Fraud or Not'])
        lst1 = {0 :'Credible Transaction' , 1 :'Fraudulent Transaction' }
        pred = pred.map(lst1)
        addresses_df['Fraud or Not'] = pred['Fraud or Not']

        
#st.write(spectra_df)
