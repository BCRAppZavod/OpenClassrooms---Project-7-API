# -*- coding: utf-8 -*-

import pickle
import streamlit as st
import pandas as pd
import numpy as np
from math import log
import matplotlib.pyplot as plt
import seaborn as sns
from functions import define_pipeline, compute_default_proba, extract_important_feats, compute_threshold, predict_df, rewrite_label, display_performance, delete_elements

#app = Flask(__name__)

st.title('Streamlit API for OpenClassrooms Project 7')

#1. Loading the data:
#DATA_PATH = "/Users/benjamincornurota/JupyterLab/OpenClassrooms/Project 7/beta_app/loaners_data_csv.txt"
DATA_PATH = https://github.com/BCRAppZavod/OpenClassrooms---Project-7-API/blob/b113ed2607fc4eb943e8709e8405a08bee4be7ea/loaners_data_csv.txt

#1.1 Extraction with Pickle:
loaners_data_csv_file = open("loaners_data_csv.txt", "rb")
loaners_data = pickle.load(loaners_data_csv_file)
loaners_data_csv_file.close()

#1.2 Selecting the number of loaners:
@st.cache
def load_data(nrows):
    data = pd.read_csv(loaners_data, nrows=nrows)
    data.dropna(inplace=True)
    return data

# Load 10,000 rows of data into the dataframe.
loaners_sample = load_data(10000)
loaners_sample = loaners_sample.drop(columns=['Unnamed: 0', 'WEEKDAY_APPR_PROCESS_START',
                                          'HOUR_APPR_PROCESS_START'])

st.subheader('This application allows you to choose between two hypotheses:')
st.write(' - The Acceptance Rate, i-e the maximum weight of False Negative')
st.write(' - The Interest Rate, i-e the long-term return on investment')

#2. Preparing the portfolio DataFrame:
#2.1 Chosing the data and the target:
X = loaners_sample.iloc[:,:-1]
y = loaners_sample.iloc[:,-1:]
y = y.to_numpy().flatten()
    
#2.2 Excluding 'SK_ID_CURR' column and saving it as a separated DF:
ID_loaners = pd.Series(X['SK_ID_CURR']).reset_index()
X = X.iloc[:,1:]

#2.3 Create optimized classifier with RandomForest:
opt_rf_pipe = define_pipeline(X, y)

#2.4 Compute Default Probability with defined Pipeline:
default_pbt_df = compute_default_proba(opt_rf_pipe, X, y, ID_loaners)

#2.5 Extract the most importance features according to the chosen classifier:
top_10_feats = extract_important_feats(opt_rf_pipe, X, y, 15)

#2.6 Create an 'Acceptance Rate' slider (from 0% to 100%):
acceptance_to_filter = st.select_slider(
     'Select an acceptance rate',
     options=[round(i, 1) for i in np.linspace(0, 1, 11)])
accept_rate_text = "{}%".format(round(acceptance_to_filter*100,2))
st.write('Acceptance Rate:', accept_rate_text)

#2.7 Compute the probability threshold according to the acceptance rate:
threshold = compute_threshold(default_pbt_df['Default_Probability'], acceptance_to_filter)
threshold_text = "{}".format(round(threshold,4))
st.write('Threshold - (Maximum Probability):', threshold_text)

#2.8 Create a portfolio with prediction results for each loaner and feature importance:
loan_amt = X['AMT_CREDIT'].values
preds_df = predict_df(threshold, default_pbt_df, y, loan_amt)
preds_df = pd.concat([preds_df, X[top_10_feats]], axis=1)
preds_df.dropna(inplace=True)
new_col_names = [rewrite_label(col) for col in preds_df.columns]
preds_df.columns = new_col_names

#3. Performance analysis:

#3.1 Macro-analysis on portfolio global performance:

#a. Selecting Interest Rate:
interest_to_filter = st.select_slider(
     'Select an interest rate hypothesis',
     options=[0.03, 0.05, 0.07])
interest_rate_text = "{}%".format(round(interest_to_filter*100,2))
st.write('Long-term Interest Rate:', interest_rate_text)

st.subheader('Macro-analysis - Portfolio Global Performance :')

#b. Extracting scores:
scores = display_performance(preds_df, interest_to_filter)
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy: ", scores["Accuracy"])
col2.metric("AUC: ", scores["AUC"])
col3.metric("Bad Rate: ", scores["Bad_Rate"])
col1, col2 = st.columns(2)
col1.metric("Number of Loans Accepted: ", int(scores["Total_Loans_Accepted"]))
col2.metric("Mean amount of loans (k$): ", int(scores["Mean_Amount"]/10e3))
col1, col2 = st.columns(2)
col1.metric("Expected Loss if Default (M$): ", int(scores["Expected_Loss"]/10e6))
col2.metric("Cost of Missed Opportunity (M$): ", int(scores["Missed_Opportunity"]/10e6))
col1, col2 = st.columns(2)
col1.metric("Difference between Expected Loss and Missed Gain (M$):", int(scores["Delta"]/10e6))
if scores["Delta"] >= 0:
    st.write("Delta is positive, i-e potential loss does not exceed missed gains")
else:
    st.write("Delta is negative, i-e potential loss are excessive.")
col2.metric("The Net Gain (M$):", int(scores["Net_Gain"]/10e6))

#c. Saving 'predictions' DF with 'session_state' to avoid rerun:
if 'preds_df' not in st.session_state:
    st.session_state['preds_df'] = preds_df

#3.2 Micro-analysis on selected potential loaner:
st.subheader('Micro-analysis - Potential Loaner Status and Main Features :')

#a. ID selection:
option_id = st.selectbox(
    'Which potential loaner do you like to see?',
     st.session_state.preds_df["Loan Id"].values)

'You selected the potential loaner with ID: ', option_id

#b. Display Status:
id_mask = st.session_state.preds_df['Loan Id'] == option_id
predicted_status = st.session_state.preds_df[id_mask]['Predicted Loan Status']
true_status = st.session_state.preds_df[id_mask]['True Loan Status']
default_pbt = st.session_state.preds_df[id_mask]['Default Probability']
col1, col2, col3 = st.columns(3)
col1.metric("Loaner predicted status: ", int(predicted_status))
col2.metric("Loaner actual status: ", int(true_status))
col3.metric("Default probability: ", round(default_pbt,3))

#c. Features selection:
all_feats = list(st.session_state.preds_df.columns)
available_feats = delete_elements(all_feats, ['Loan Id','Predicted Loan Status','True Loan Status', 'Threshold','Default Probability'])
#Feat 1:
option_feat = st.selectbox(
    'Select your first feature:',
     available_feats)

'You selected the feature: ', option_feat
#Feat 2:
available_feats_2 = available_feats.copy()
available_feats_2.remove(option_feat)
option_feat_2 = st.selectbox(
    'Select your second feature:',
     available_feats_2)

'You selected the feature: ', option_feat_2

#d. Display DataFrame:
st.dataframe(st.session_state.preds_df[id_mask][[option_feat, option_feat_2]])

#e. Display Plot:
#Function for one feature:
def draw_hist(df, selected_feat):
    """Draw a histogram plot and apply the Sturges formula to compute the number of bins."""
    bins = round((1 + log(len(df[selected_feat]),2)))
    fig = plt.figure(figsize=(9,6))
    sns.histplot(data=df, x=selected_feat, bins=bins)
    plt.xlabel(rewrite_label(selected_feat))
    plt.xticks(rotation=35)
    st.pyplot(fig)

#Function for two features:   
def choose_chart_type(df, selected_feat, selected_feat_2):
    """In case of a bivariate analysis, there are 4 possibilities:
    1) The two variables are discrete, here less than 10 unique values.
    2) The two are continuous, here more than 10 unique values.
    3) X is discrete and Y is continuous.
    4) X is continuous and Y is discrete.
    This function automatically returns the proper plot type that fits the best the data according to these 4 possible outcomes."""
    num_values_1 = len(df[selected_feat].unique())
    num_values_2 = len(df[selected_feat_2].unique())
    
    if num_values_1 <= 10 and num_values_2 <= 10:
        st.write('Both discrete variables')
        fig = plt.figure(figsize=(9,6))
        sns.countplot(x=selected_feat, hue=selected_feat_2, data=df)
        plt.xticks(rotation=35)
        st.pyplot(fig)
    elif num_values_1 >= 10 and num_values_2 <= 10:
        st.write('Feat 1 is continuous, Feat 2 is discrete')
        fig = plt.figure(figsize=(9,6))
        sns.boxplot(x=selected_feat, y=selected_feat_2, data=df, orient="h")
        plt.xticks(rotation=35)
        st.pyplot(fig)
    elif num_values_1 <= 10 and num_values_2 >= 10:
        st.write('Feat 1 is discrete, Feat 2 is continuous')
        fig = plt.figure(figsize=(9,6))
        sns.boxplot(x=selected_feat, y=selected_feat_2, data=df)
        plt.xticks(rotation=35)
        st.pyplot(fig)
    else:
        st.write('Both continuous variables')
        fig = plt.figure(figsize=(9,6))
        sns.lineplot(x=selected_feat, y=selected_feat_2, data=df)
        plt.xticks(rotation=35)
        st.pyplot(fig)    

#Choose to plot one or two variables:
page_names = ['Univariate','Bivariate']

page = st.radio("Select 'Univariate' to display a one-feature chart, else select 'Bivariate' to display a two-feature chart", page_names)

if page == 'Univariate':
    if st.session_state.preds_df[option_feat].dtype != 'object':
        #If values are numerical:
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean: ", round(st.session_state.preds_df[option_feat].mean(), 2))
        col2.metric("Median: ", round(st.session_state.preds_df[option_feat].median(), 2))
        col3.metric("Standard Deviation: ", round(st.session_state.preds_df[option_feat].std(), 2))
        draw_hist(st.session_state.preds_df, option_feat)
    else:
        #If values are categorical:
        draw_hist(st.session_state.preds_df, option_feat)
    
else:
    choose_chart_type(st.session_state.preds_df, option_feat, option_feat_2)
