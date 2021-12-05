# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 09:35:59 2021

@author: Menuka_08214
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data=pd.read_csv("input2.csv")
sns.countplot(x ='y',data=data,palette='hls')
plt.show()

count_no_sub = len(data[data['y']==0])
count_sub = len(data[data['y']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of Disconnections =", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of Connections =", pct_of_sub*100)

#
#data.A.hist()
#plt.title('Histogram of Age(Wives)')
#plt.xlabel('Age')
#plt.ylabel('Frequency')
#plt.savefig('hist_age')


#Ont Model vs Billing status
#table=pd.crosstab(data.ONT,data.y)
#table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
#plt.title("Stacked Bar - Ont Model vs Billing status")
#plt.xlabel("Ont Model")
#plt.ylabel('Billing status')
#plt.savefig('edu_vs_pur_stack')

#Total ONT count per port vs Billing status
#table=pd.crosstab(data.ONTperport,data.y)
#table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
#plt.title("Stacked Barchart - Total ONT count per port vs Billing status")
#plt.xlabel("Total ONT count per port")
#plt.ylabel('Proportion of connection status')
#plt.savefig('edu_vs_pur_stack')


#data.Distance.hist()
#plt.title('Histogram of Distances')
#plt.xlabel('Distance')
#plt.ylabel('Frequency')
##plt.savefig('hist_age')


#table=pd.crosstab(data.RX,data.y)
#table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
#plt.title("Stacked Barchart - Total ONT count per port vs Billing status")
#plt.xlabel("Total ONT count per port")
#plt.ylabel('Proportion of connection status')
#plt.savefig('edu_vs_pur_stack')


