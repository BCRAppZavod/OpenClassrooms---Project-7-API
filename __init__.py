#Functions used for Streamlit API "Solvency Prediction" of OpenClassrooms Project n°7

import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def define_pipeline(X_matrix, target):
    """Return a Pipeline with following steps: 
    1) Create a preprocessor for numerical and categorical features
    2) Oversampling the minority class (loaners in default) and undersampling the majority
    3) The chosen classifier (here RandomForest) with optimized hyperparameters"""
    #1. Selecting Features Class:
    categ_feats = make_column_selector(dtype_exclude=np.number)
    num_feats = make_column_selector(dtype_include=np.number)
    
    #2. Creation of a transformation pipeline:
    numerical_pipeline = make_pipeline(RobustScaler())
    categorical_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))

    #3. Creation of the final features preprocessor:
    preprocessor = make_column_transformer((categorical_pipeline, categ_feats),  (numerical_pipeline, num_feats))
    
    #4. Defining re-sampling strategy:
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)

    #5. Chosen model - Optimized random forest classifier:
    m_name = "RandomForest"
    rs = 42
    opt_RF = RandomForestClassifier(max_depth=30, max_features='auto', n_estimators=200, random_state=rs)
    
    #6. Creation of a pipeline:
    pipe_steps = [('preprocessor', preprocessor),('oversampling', over),('undersampling', under),('RF', opt_RF)]
    opt_rf_pipe = Pipeline(steps=pipe_steps)
    return opt_rf_pipe

def compute_default_proba(pipeline, X_matrix, target, loaners_id):
    """Predict probabilities of default for each loaner with its ID:"""
    pipeline.fit(X_matrix, target)
    default_pbt = pipeline.predict_proba(X_matrix)[:, 1]
    return pd.DataFrame(zip(loaners_id['SK_ID_CURR'], default_pbt), columns=['Loan_ID','Default_Probability'])


def drop_duplicated_feats(important_feats, original_feats):
    saved_feats = list()
    for imp_feat in important_feats:
        for orig_feat in original_feats:
            if orig_feat in imp_feat:
                saved_feats.append(orig_feat)
    return list(pd.Series(saved_feats).drop_duplicates().values)

def extract_important_feats(pipeline, X_matrix, target, n_feats):
    """Extract the most important features according to the chosen classifier:
    1) First, it is necessary to retrieve their original names.
    2) Some features are identical due to 'OneHotEncoder' which transforms the categorical columns into a sparse matrix, therefore, duplicates must be deleted."""
    #1. Extract original features name:
    feats_name = list()
    feats_name.extend(list(pipeline.steps[0][1].transformers_[0][1]['onehotencoder'].get_feature_names_out()))
    feats_name.extend(list(pipeline.steps[0][1].transformers_[1][1]['robustscaler'].feature_names_in_))

    #2. Compute important features:
    pipeline.fit(X_matrix, target)
    feat_importance = pd.Series(pipeline.steps[3][1].feature_importances_,index=feats_name)
    top_n_feats = list(feat_importance.nlargest(n_feats).index)

    #3. Delete duplicated features:
    top_n_feats = drop_duplicated_feats(top_n_feats, X_matrix.columns)
    return top_n_feats

def compute_threshold(default_pbt, acceptance):
    """Returns the threshold (maximum probability) according to a given acceptance rate, 
    i-e what percentage of new loans are accepted to keep the number of defaults low."""
    return np.quantile(default_pbt, acceptance)

def predict_df(threshold, default_pbt, true_targets, loan_amt):
    """Returns a table with the predicted status for each loaner according to their 
    default probability given by the chosen model and the threshold."""
    thres_col = [round(threshold,3) for i in range(default_pbt.shape[0])]
    default_pbt['Threshold'] = thres_col
    default_pbt['Predicted_Loan_Status'] = default_pbt['Default_Probability'].apply(lambda x: 1 if x > threshold else 0)
    default_pbt['True_Loan_Status'] = true_targets
    default_pbt['Loan_Amount'] = loan_amt
    return default_pbt

def rewrite_label(lab):
    if '_' in lab:
        lab = " ".join([wd.capitalize() for wd in lab.split('_')])
    if 'Amt' in lab :
        lab = "Amount of".join([wd.capitalize() for wd in lab.split('Amt')])
    if 'Num' in lab :
        lab = "Number of".join([wd.capitalize() for wd in lab.split('Num')])
    return lab

def display_performance(preds, interest_rate):
    """Compute key-scores and global performance, such as net loss risk, net gain, the cost of missed opportunities on the chosen portfolio"""
    true_targets = preds['True Loan Status']
    predicted_results = preds['Predicted Loan Status']
    acc = round(accuracy_score(true_targets, predicted_results),2)
    auc = round(roc_auc_score(true_targets, predicted_results),2)
    
    # Compute FN and FP:
    tn, fp, fn, tp = confusion_matrix(true_targets, predicted_results).ravel()
    
    # Compute Bad rate:
    accepted_loans = preds[preds['Predicted Loan Status'] == 0]
    bad_rate = np.sum((accepted_loans['True Loan Status']) / accepted_loans['True Loan Status'].count())
    bad_rate = round(bad_rate*100,2)
    
    #Compute mean loan amount:
    mean_amt = round(accepted_loans['Loan Amount'].mean())
    
    #Portfolio expected loss:
    exp_loss = fn * mean_amt
    
    #Portfolio missed opportunity according to the hypothetic interest rate:
    missed_gain = fp * mean_amt * interest_rate
    
    #Compute 'Delta' between net loss ans missed opportunity:
    delta = missed_gain - exp_loss
    
    #Portfolio net gain:
    total_accepted = accepted_loans['Predicted Loan Status'].count()
    net_gain = total_accepted * mean_amt * interest_rate - exp_loss
    
    return {"Accuracy": acc, "AUC": auc, "Bad_Rate": bad_rate, "Mean_Amount": mean_amt, "Total_Loans_Accepted": total_accepted, "Expected_Loss": exp_loss, "Missed_Opportunity": missed_gain, "Delta": delta, "Net_Gain": net_gain}
    
def delete_elements(original_list, feats_to_del):
    for feat in feats_to_del:
        original_list.remove(feat)
    return original_list