# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
#------------------- import splitting functions-------------------
from sklearn.model_selection import train_test_split

import pandas as pd

# +
# ------------------- Split dataset -------------------


def split_function(df, target_varible):
    """
    The function split_data splits the original DataFrame df into training, validate and 
    test sets using the train_test_split function from the library Scikit-Learn(machine learning library in
    Python that provides tools for data preprocessing, model selection, training and evaluation).
    """
    train, test = train_test_split(df,
                                   random_state=123,
                                   test_size=.20,
                                   stratify= df[target_varible])
    
    train, validate = train_test_split(train,
                                   random_state=123,
                                   test_size=.25,
                                   stratify= train[target_varible])
    return train, validate, test

# +
# ------------------- Iris dataset -------------------

def prep_iris(df):
    """
    The function prep_iris removes unnecessary columns and renames the species column 
    for preprocessing the Iris dataset.
    """
    df = df.drop(columns= (['species_id', 'measurement_id']))
    df = df.rename(columns = {'species_name': 'species'})
    return df



# +
# ------------------- Titanic dataset -------------------

def prep_titanic(titanic):
    """
    The function prep_titanic removes specific columns ('class', 'embarked', 'deck', 'age') and drops rows with 
    missing values to preprocess the Titanic dataset, likely for analysis or modeling purposes.
    """
    titanic = titanic.drop(columns=['class', 'embarked', 'deck', 'age'])
    titanic = titanic.dropna()
    return titanic



# +

# ------------------- Titanic dataset prepping for decision tree and other algorithms -------------------

def prep_titanic_for_dt(df):
    '''
    The function prep_titanic_for_dt removes specific columns ('class', 'embarked', 'deck','passenger_id',
    'embark_town') and drops any duplicates to preprocess the Titanic dataset,
    likely for analysis or modeling purposes.
    '''
    
    #drop any duplicates
    df = df.drop_duplicates()
    
    
    #encoding categorical variables in a format suitable for machine learning algorithms
    # Perform one-hot encoding(creating multiple binary columns) for 'embark_town' column
    #creates three new binary columns,1 and 0 values in these new columns indicate 
    #whether a passenger embarked at that specific town or not.
    embark_dummies = pd.get_dummies(df['embark_town'], drop_first=False, dtype='int')
    df = pd.concat([df, embark_dummies], axis=1)
    

    
    #encoding categorical variables in a format suitable for machine learning algorithms
    # Create dummy variables for 'sex' column and drop one of the dummies (keeping only 'sex' column)
    # Changes the 'sex' column to have 1 for 'female' and 0 for 'male'
    #  dummy_df = pd.get_dummies(df['sex'], drop_first=True, dtype='int')
    #  df['sex'] = dummy_df
    
    # Create sex dummy variables
    sex_dummies = pd.get_dummies(df['sex'], drop_first=False, dtype='int')
    # Concatenate sex dummy variables to the DataFrame
    df = pd.concat([df, sex_dummies], axis=1)
    # Drop the 'sex' column
    df.drop('sex', axis=1, inplace=True)

    
    
    # Convert 'age' column to integers after handling NaN values
    df['age'] = df['age'].fillna(0).astype(int)  # Replace NaN with 0 and convert to int

    
    
    # Drop specified columns
    df = df.drop(columns=[ 'deck', 'embarked', 'class', 'passenger_id', 'embark_town'])
    
    
    # Return the modified DataFrame
    return df


# +
# ------------------- Telco dataset -------------------

def prep_telco(telco):
    """
    The function prep_telco function performs various data preprocessing steps on the Telco dataset, 
    including dropping columns, encoding categorical variables, creating dummy variables, 
    and converting a column to a numerical format.
    """




    telco = telco.drop(columns=['internet_service_type_id', 'contract_type_id', 'payment_type_id'])

    telco['gender_encoded'] = telco.gender.map({'Female': 1, 'Male': 0})
    telco['partner_encoded'] = telco.partner.map({'Yes': 1, 'No': 0})
    telco['dependents_encoded'] = telco.dependents.map({'Yes': 1, 'No': 0})
    telco['phone_service_encoded'] = telco.phone_service.map({'Yes': 1, 'No': 0})
    telco['paperless_billing_encoded'] = telco.paperless_billing.map({'Yes': 1, 'No': 0})
    telco['churn_encoded'] = telco.churn.map({'Yes': 1, 'No': 0})
    
    dummy_df = pd.get_dummies(telco[['multiple_lines',
                                     'online_security',
                                     'online_backup',
                                     'device_protection', 
                                     'tech_support',
                                     'streaming_tv',
                                     'streaming_movies', 
                                     'contract_type', 
                                     'internet_service_type',
                                     'payment_type']],
                                  drop_first=True)
    
    telco = pd.concat( [telco, dummy_df], axis=1 )
    
    telco.total_charges = telco.total_charges.str.replace(' ', '0').astype(float)
    
    return telco


# +
# ------------------- Telco dataset prepping for decision tree -------------------

def prep_telco_for_dt(df_telco):
    """
    The function prep_telco function performs various data preprocessing steps on the Telco dataset, 
    including dropping columns, converting churn column to a numerical format, converting
    total_charges column to a float and cleaning total_charges .
    
    """


    # drop any duplicates
    df_telco = df_telco.drop_duplicates()

    # Drop specified columns
    df_telco = df_telco.drop(columns = ['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents', 
                                          'phone_service', 'multiple_lines','online_security', 'online_backup', 
                                          'device_protection','tech_support','streaming_tv','streaming_movies',
                                          'paperless_billing', 'contract_type', 'internet_service_type', 
                                          'payment_type','internet_service_type_id', 'contract_type_id', 
                                          'payment_type_id'])
                                   
                                  
    # Remove leading and trailing spaces from 'total_charges' column
    df_telco['total_charges'] = df_telco['total_charges'].str.strip()

    # Remove rows where 'total_charges' is empty
    df_telco = df_telco[df_telco.total_charges != '']

    # Convert 'total_charges' column to float
    df_telco['total_charges'] = df_telco['total_charges'].astype(float)

    # Encoding the target variable 'churn' as 1 for 'Yes' and 0 for 'No' 
    df_telco['churn'] = df_telco['churn'].map({'Yes': 1, 'No': 0})

    # Convert 'churn' column to integer
    df_telco['churn'] = df_telco['churn'].astype(int)
    
    return df_telco
# -


