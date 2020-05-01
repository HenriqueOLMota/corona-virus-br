# -*- coding: utf-8 -*-
#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import kaggle
import numpy as np
#%%
# Authenticate API and download the CSV file
kaggle.api.authenticate()
kaggle.api.dataset_download_files('unanimad/corona-virus-brazil', 
                                  r'C:\Users\CepaTech\Desktop\Machine Learning\Datasets\Corona Brasil', 
                                  unzip = True)
#%%
data = pd.read_csv('brazil_covid19.csv')
#Convert elements in data['date'] to date format (with day first)
data['date'] = pd.to_datetime(data['date'],dayfirst=True)
#%% First I will plot cases and deaths by region
#Sum all cases and deaths at the same region in the same day
data_region = data.groupby(['date','region']).sum().reset_index()
#Grid on plot
sns.set_style("whitegrid")
#g->Figure instance; ax = Object (or array), indicates plot positions; size of plot
reg,regax = plt.subplots(2,figsize=(10,12))

#DateFormatter = How date will be show; Next line put axis X in the way I define
#by DateFormatter; Next line I declare space between dates = 1 week
date_form = DateFormatter("%d/%m")
regax[0].xaxis.set_major_formatter(date_form)
regax[0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
#Plot
reg1 = sns.lineplot(data=data_region,x='date',y='cases',
                   hue='region',marker='o',ax=regax[0])
regax[0].set_title(u'Evolução dos casos de COVID-19 no Brasil')
#Definig legend title
regleg1 = regax[0].legend()
regleg1.texts[0].set_text(u"Região")
regax[0].set_xlabel('Datas')
regax[0].set_ylabel(u'Nº de Casos')
#first and last ticks on X axis
regax[0].set(xlim=['2020-01-30','2020-04-30'])
fig2 = sns.lineplot(data=data_region,x='date',y='deaths',
                   hue='region',marker='o',ax=regax[1])
regax[1].xaxis.set_major_formatter(date_form)
regax[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
regax[1].set_title(u'Evolução das mortes por COVID-19 no Brasil')
regleg2 = regax[1].legend()
regleg2.texts[0].set_text(u"Região")
regax[1].set_xlabel('Datas')
regax[1].set_ylabel(u'Nº de Mortes')
regax[1].set(xlim=['2020-01-30','2020-04-30'])
#Here im rotating/aligning both X ticks and adjusting plots to not overlap 
plt.setp(regax[0].get_xticklabels(), rotation=45, ha="right")   # optional
plt.setp(regax[1].get_xticklabels(), rotation=45, ha="right")
reg.subplots_adjust(bottom=-0.2)
#%% Now I will plot cases and deaths in Brasil
data_brasil = data.groupby('date')['cases','deaths'].sum().reset_index()
data_brasil
br,brax = plt.subplots(figsize=(12,8))
br1 = sns.lineplot(data=data_brasil,x='date',y='cases',
                   marker='o',ax=brax,label='Casos')
br2 = sns.lineplot(data=data_brasil,x='date',y='deaths',
                   marker='o',ax=brax,label='Mortes')
br.autofmt_xdate()
brax.set_title(u'Evolução de casos e mortes por COVID-19 no Brasil',
               fontsize='x-large')
brax.set_xlabel('Data')
brax.set_ylabel(u'Número de casos / mortes')
brax.legend(fontsize='x-large')
brax.set(xlim=['2020-01-30','2020-04-29'])
brax.xaxis.set_major_formatter(date_form)
brax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
#%% Plot Brasil cases, but now in logscale (using index in X axis)
logbr,logbrax = plt.subplots(figsize=(12,8))
logbr1 = sns.lineplot(data=data_brasil,x='date',y='cases',
                   marker='o',ax=logbrax,label='Casos')
logbrax.set(yscale="log")
logbrax.legend(fontsize='x-large')
logbrax.set_xlabel('Data',fontsize='x-large')
logbrax.set_ylabel('Casos',fontsize='x-large')
logbrax.set_title(u'Evolução de casos por COVID-19 no Brasil (escala log)',
               fontsize='x-large')
#%% Plot new cases
data_brasil['newCases'] = data_brasil['cases'].diff().fillna(0)
nc,ncax = plt.subplots(figsize=(12,8))
                       
nc1 = sns.lineplot(data=data_brasil,x='date',y='newCases',marker='o',
                   ax = ncax, label='Novos Casos')
nc.autofmt_xdate()
ncax.set_title(u'Novos casos de COVID-19 no Brasil',
               fontsize='x-large')
ncax.set_xlabel('Data')
ncax.set_ylabel(u'Número de casos')
ncax.legend(fontsize='x-large')
ncax.set(xlim=['2020-01-30','2020-04-29'])
ncax.xaxis.set_major_formatter(date_form)
ncax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
#%%