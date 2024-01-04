# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 21:41:36 2023

@author: ahmed.abdelmonim
"""

import pandas as pd
import numpy as np

Demo = pd.read_csv("Emp_Demo.csv")
Attrition  = pd.read_csv("Emp_Attrition.csv")
Income  = pd.read_csv("Emp_Income.csv")
Job_Details = pd.read_csv("Emp_Job Details.csv")

# merge data sets
from functools import reduce
dfs = [Demo,Attrition,Income,Job_Details]

df = reduce (lambda left,right:pd.merge(left,right,on=['EmpId'],how = 'outer'),dfs)
df.drop(['Department_y','StandardHours_y'], axis = 1)

# checking data features 
df.head()
# data dimension 
df.shape

# names of variables 
df.columns

# checking structure of the data
print(df.info())

#checking summery of the data 
print(df.describe(include = 'all'))


# checking of Missing values 

print(df.isnull().sum())


# handling missing values
df = df.dropna()

# Add age groups to df


conditions = [

     (df['Age'] >= 18)& (df['Age'] <= 24),
    (df['Age'] > 24)& (df['Age'] <= 34),
    (df['Age'] > 34) & (df['Age'] <= 44),
    (df['Age'] > 44)& (df['Age'] <= 54),
    (df['Age'] > 54)& (df['Age'] <= 60)

   ]
values = ["18-24", "25-34", "35-44","45-54", "55-60"]
df['Age_group'] = np.select(conditions, values)


#Add income groups to df
df['MonthlyIncome'] = df['MonthlyIncome'].astype('int')
conditions_1 = [

     (df['MonthlyIncome'] >= 1000)& (df['MonthlyIncome'] <= 2000),
    (df['MonthlyIncome'] > 2000)& (df['MonthlyIncome'] <= 4000),
    (df['MonthlyIncome'] > 4000) & (df['MonthlyIncome'] <= 6000),
    (df['MonthlyIncome'] > 6000)& (df['MonthlyIncome'] <= 8000),
    (df['MonthlyIncome'] > 8000)& (df['MonthlyIncome'] <= 12000),
    (df['MonthlyIncome'] > 12000)& (df['MonthlyIncome'] <= 15000),
    (df['MonthlyIncome'] > 15000)
   ]
values_1 = ["1000-2000", "2001-4000", "4001-6000",
                            "6001-8000", "8001-12000","12001-15000",">15000"]
df['Income_group'] = np.select(conditions_1, values_1)

df.to_csv('HR_1.csv')






















