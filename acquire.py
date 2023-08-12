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

import os
import pandas as pd
import numpy as np
#from env import get_db_url
import env
from sqlalchemy import create_engine, text

##############################################################################
def get_db_connection(database):
    return get_connection(database)

##############################################################################
'''
#Functions for the first part of Data Acquisition exercises 

# Get titanic data 
def get_titanic_data():
    # Create SQL query.
    sql_query = 'SELECT * FROM passengers;'
    # Read in dataframe from Codeup db
    df = pd.read_sql(sql_query, env.get_db_url(env.user,env.host,env.password, 'titanic_db'))
    return df


# Get iris data 
def get_iris_data():
    sql_query = ("SELECT species_id, measurement_id, species_name, sepal_length, sepal_width, petal_length, petal_width FROM measurements JOIN species USING(species_id)")
    # Read in dataframe from Codeup db.
    df = pd.read_sql(sql_query, env.get_db_url(env.user,env.host,env.password,'iris_db'))
    return df

# Get telco data 
def get_telco_data():
    sql_query = ("SELECT * from customers JOIN contract_types USING (contract_type_id) JOIN internet_service_types USING (internet_service_type_id) JOIN payment_types USING (payment_type_id)")
    # Read in dataframe from Codeup db
    df = pd.read_sql(sql_query,env.get_db_url(env.user,env.host,env.password,'telco_churn'))
    return df



'''
###############################################################################
# Updated the get function for the second part of the Data Acquisition exercise
# Get titanic data 
#Adding caching


def updated_get_titanic_data():
    csv_filename = 'titanic.csv'
    
    if os.path.exists(csv_filename):
        print(f'Using cached {csv_filename}')
        return pd.read_csv(csv_filename)
    else:
        print('Acquiring data from SQL database')
        df = new_titanic_data()
        # Save to CSV for caching
        df.to_csv('titanic.csv')
    return df



def new_titanic_data():
    # Create SQL query.
    sql_query = 'SELECT * FROM passengers'
    # Read in dataframe from Codeup db
    df = pd.read_sql(sql_query, env.get_db_url(env.user,env.host,env.password, 'titanic_db'))
    return df



###############################################################################
# Updated the get function for the second part of the Data Acquisition exercise
# Get telco data 


def updated_get_telco_data():
    csv_filename = 'telco.csv'
    
    if os.path.exists(csv_filename):
        print(f'Using cached {csv_filename}')
        return pd.read_csv(csv_filename)
    else:
        print('Acquiring data from SQL database')
        df = new_telco_data()
        # Save to CSV for caching
        df.to_csv('telco.csv')
    return df
    

def new_telco_data():
    sql_query = ("SELECT * from customers JOIN contract_types USING (contract_type_id) JOIN internet_service_types USING (internet_service_type_id) JOIN payment_types USING (payment_type_id)")
    # Read in dataframe from Codeup db
    df = pd.read_sql(sql_query,env.get_db_url(env.user,env.host,env.password,'telco_churn'))
    return df
    

###############################################################################
# Updated the get function for the second part of the Data Acquisition exercise
# Get iris data 


def updated_get_iris_data():
    csv_filename = 'iris.csv'
    
    if os.path.exists(csv_filename):
        print(f'Using cached {csv_filename}')
        return pd.read_csv(csv_filename)
    else:
        print('Acquiring data from SQL database')
        df = new_iris_data()
        # Save to CSV for caching
        df.to_csv('iris.csv')
    return df


def new_iris_data():
    sql_query = ("SELECT species_id, measurement_id, species_name, sepal_length, sepal_width, petal_length, petal_width FROM measurements JOIN species USING(species_id)")
    # Read in dataframe from Codeup db.
    df = pd.read_sql(sql_query, env.get_db_url(env.user,env.host,env.password,'iris_db'))
    return df
###############################################################################
# acquire employees
# is a modularized script
# ready to acquire the first 100 rows of employee 
# data for an end user
# this requires a specific env file setup

# make a funcion that reads content in from sql
def grab_data(
    db,
    user=env.user,
    password=env.password,
    host=env.host):
    '''
    grab data will query data from a specified positional argument (string literal)
    schema from an assumed user, password, and host provided
    that they were imported from an env
    
    return: a pandas dataframe
    '''
    query = '''SELECT * FROM employees LIMIT 100'''
    connection = f'mysql+pymysql://{user}:{password}@{host}/{db}'
    df = pd.read_sql(query, connection)
    return df

def acquire_100_emps(
        file_loc='employees_first_100.csv'):
    '''
    acquire_100_emps will check the path at file_loc
    in order to see if a csv (cached version of data)
    exists on the user's computer, if not, it will use the grab_data
    function to query the information from the employees schema.
    
    this is formatted to work with a specific env file structure
    and is inteded to work with the codeup cloud server.
    please reference docs at readme.md at github.com/fakenotreal
    
    return: a single pandas dataframe consisting of the first 100
    rows of employee data
    '''
    if os.path.exists(file_loc):
        df = pd.read_csv(file_loc, index_col=0)
    else:
        # read it in from sql!
        df = grab_data('employees')
        # if we grabbed the data from sql once, cache it so we 
        # dont need to depend
        # on the server every time
        df.to_csv(file_loc, index=False)
    return df
# -


