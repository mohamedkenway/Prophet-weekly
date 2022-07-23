import os
import sys
import pandas as pd
from prophet import Prophet
from hijri_converter import Gregorian 
from sklearn.metrics import mean_absolute_percentage_error
from datetime import timedelta,datetime
import numpy as np
def get_holiday(dates):
    holidays=[]
    for date in dates:
        days = [date + timedelta(days=i) for i in range(7)]
        fh=["01-01","01-10","03-12","10-01","10-02","12-10","12-11"]
        fg=["01-01","01-12","05-01","07-05","11-01"]
        table=[]
        for i in days:
            bl=False
            hijridate=Gregorian.fromisoformat(i.strftime('%Y-%m-%d')).to_hijri().isoformat()
            for j in fg:
                if(j in str(i.strftime('%Y-%m-%d'))):
                    bl=True 
            for j in fh:
                if(j in hijridate):
                    bl=True
            if bl:
                table.append(1)
            else :
                table.append(0)
        holidays.append(sum(table)/7)
    return holidays
def get_hijrimm(dates):
    hijrimm=[]
    for date in dates:
        days = [date + timedelta(days=i) for i in range(7)]
        table=[]
        for i in days:
            hijridate=Gregorian.fromisoformat(i.strftime('%Y-%m-%d')).to_hijri().isoformat()
            hijri=hijridate.split('-')
            if hijri[1]==8 or hijri[1]==9 or hijri[1]==12:
                table.append(1)
        hijrimm.append(sum(table)/7)
    return hijrimm
def fbprophet(data):
    train=data[:len(data['Quantité'])-1]
    test=data['Quantité'][len(data['Quantité'])-1:]
    train=train.rename(columns={'week':'ds','Quantité':'y'})
    train.reset_index(inplace=True)
    train=train[["ds","y",'ferie','hijrimm']]
    model=Prophet()
    model.add_seasonality(name="weekly",period=52 ,fourier_order=3)
    #adding regressors
    model.add_regressor('ferie')
    model.add_regressor('hijrimm')
    model.fit(train)
    future = model.make_future_dataframe(periods=1, freq='W-wed')
    future['ferie']=get_holiday(future['ds'])
    future['hijrimm']=get_hijrimm(future['ds'])
    forcast =model.predict(future)
    fs=forcast['yhat'][len(train):]
    return (['Prophet',mean_absolute_percentage_error(test,fs),fs])
def predict(d):
        d.index=pd.DatetimeIndex(d['week'])
        d=d.resample("W-WED").last()
        d=d.reset_index(drop=True)
        fb=fbprophet(d)
        return (fb)
if __name__ == '__main__':
    d=pd.read_csv(os.path.join(sys.path[0],'test article.csv'))
    start=datetime.now() 
    #replace missing values
    for j,k in d.iterrows():
        if (d.loc[j,'Quantité']/d['Quantité'].mean()*100) <5: 
            d.loc[j,'Quantité']=d['Quantité'].mean()
    #sorting values by week
    d.sort_values(by='week',inplace=True) 
    d.index=pd.DatetimeIndex(d['week'])
    #making prediction
    p=predict(d)
    end=datetime.now()
    print(end-start)
    dff=pd.DataFrame({'prediction':p[2].tolist()[0],'MAPE':p[1]})
    print(dff)
