import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------
# DATA HANDLING
# ----------------------------------------------------------------------------------------------

# DATA TYPE, AND NA CALCULATOR
def na_calculator(data):
    '''SHOWING THE STATS OF MISSING DATA AND DATA TYPE'''

    percentage_missing = np.around(data.isna().mean()*100, 2).sort_values(ascending = False)                      # Storing the Percentages of NaNs
    sum_missing = data.isna().sum().sort_values(ascending = False)                                    # Storing the Sum of NaNs
    names = sum_missing.index.to_list()                                                               # Storing names (to show in the columns)
    data_type = data[names].dtypes                                                                    # Storing the type of data based on the order from the previous obtained data (slicing)
    sum_values = sum_missing.to_list()                                                                # Getting count of missing values
    perc_values = np.around(percentage_missing.to_list(), 3)                                          # Getting percentage of missing values
    types = data_type.to_list()                                                                       # Getting the types of the data
    
    # TURN ALL THE DATA INTO A DATAFRAME
    df_missing = pd.DataFrame({"NAMES" : names,
                                    "VALUE COUNT" : sum_values,
                                    "PERCENTAGE (%)" : perc_values,
                                    "DATA TYPE": types})
    return df_missing

# PASRING STOCK COMPANY CODES INTO FULL NAMES
def get_company_name(ticker_symbol):
    '''RETRIEVES THE FULL COMPANY NAME FOR A GIVEN STOCK TICKER SYMBOL'''
    try:
        ticker          = yf.Ticker(ticker_symbol)
        info            = ticker.info
        company_name    = info.get('longName')

        if(company_name):
            return company_name
        else:
            return 'Unknown'
    except Exception as e:
        return f'Error: {e}'







