import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from statsmodels.tsa.stattools import coint

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def get_comp_list(pricesdf,sic_list,since):
    prices_df = pricesdf#.dropna()
    prices_df['date'] = pd.to_datetime(prices_df['date'],format="%Y%m%d").dt.to_period('M')
    prices_df['SICCD'] = prices_df['SICCD'].astype(str)
    prices_df['SICCD'] = prices_df['SICCD'].astype(str).apply(lambda x:x.split('.')[0])

    #prices_df = prices_df[prices_df['SICCD'].isin(sic_list)]#filter by sic number
    prices_df = prices_df[prices_df['EXCHCD'].isin([1.0,2.0,3.0])]#filter by exchange code
    prices_df = prices_df[prices_df['TRDSTAT']=='A']#if it is trading or not
    #removed null
    prices_df = prices_df[prices_df['PRC']>0]#check if prices were not negative
    prices_df['MARKCAP'] = prices_df['PRC']*prices_df['SHROUT']#calcualte market cap
    prices_df = prices_df[prices_df['date']>since]
    prices_df = prices_df.sort_values(by='date')
    grp_obj = prices_df.groupby(['date','TICKER'])
    tick_00 = [key[1] for key in grp_obj.groups.keys() if str(key[0])[:4] == '2000']
    tick_20 = [key[1] for key in grp_obj.groups.keys() if str(key[0])[:4] == '2019']
    comp_list = intersection(tick_20, tick_00)
    prices_df = prices_df[prices_df['TICKER'].isin(comp_list)]
    prices_df = prices_df[prices_df['SICCD'].astype(str).isin(sic_list)]
    #prices_df = prices_df.groupby('date').get_group(pd.Period('2000-01', 'M'))
    return prices_df

def get_ratios(funda_df,comp_list,alpha,period,gvkey):
    ratios_df = funda_df[funda_df['tic'].isin(comp_list)]
    ratios_df['datadate'] = pd.to_datetime(ratios_df['datadate'],format="%Y%m%d").dt.to_period('M')
    ratios_df['rd/rev'] = ratios_df['xrdq']/ratios_df['revtq']
    ratios_df['roe'] = ratios_df['niq']/(ratios_df['atq']-ratios_df['ltq'])
    ratios_df['p/e'] = ratios_df['prccq']/ratios_df['epsfxq']
    ratios_grp_obj = ratios_df[['datadate','tic','prccq','roe','rd/rev','p/e','epsfxq']].groupby('tic')

    temp_list = []
    for tic in ratios_grp_obj.groups.keys():
        temp_df = ratios_grp_obj.get_group(tic).set_index('datadate')
        ewm = temp_df.ewm(alpha=alpha,min_periods=period).mean()
        ewm.insert(0,'tic',tic)
        ewm.insert(1,'real roe',temp_df['roe'])
        ewm.insert(3,'real rd',temp_df['rd/rev'])
        ewm.reset_index(inplace=True)
        temp_list.append(ewm)
    ratios_df = pd.concat(temp_list)
    ratios_df = ratios_df[ratios_df['datadate'].dt.month==12]
    ratios_df['datadate'] = ratios_df['datadate'].apply(lambda x:int(str(x)[:4]))
    ratios_df = ratios_df.rename({'tic':gvkey},axis=1)
    ratios_df = ratios_df.set_index(['datadate',gvkey])
    return ratios_df

def sim_df(prices_df,gvkey2018,year,sim_scores):
    grp_obj = prices_df.groupby('date')
    tic = grp_obj.get_group(pd.Period(str(year)+'-01', 'M')).sort_values('MARKCAP',ascending=False).head(20)['TICKER'].unique()
    gvkey2018['sic'] = gvkey2018['sic'].astype(str)
    gvkey2018 = gvkey2018[gvkey2018['tic'].isin(tic)]
    gvkey_tic_dict = dict(gvkey2018[['gvkey','tic']].values)
    gvkey_tic_dict.keys()
    temp_df = sim_scores.groupby('year').get_group(year)
    temp_df = temp_df[(temp_df['gvkey1'].isin(gvkey_tic_dict.keys())) & (temp_df['gvkey2'].isin(gvkey_tic_dict.keys()))]
    sim_df = temp_df.replace({'gvkey1':gvkey_tic_dict,'gvkey2':gvkey_tic_dict}).sort_values(by='score',ascending=False)
    sim_df = sim_df.groupby('score').tail(1)
    return sim_df

def merge_factors(colname,ratios_df,sim_df):
    df = ratios_df[colname].dropna()
    melt_df = pd.DataFrame(df.values - df.values[:, None],columns=df.index,index=df.index).unstack()
    melt_df = pd.DataFrame(zip(melt_df.index.values,melt_df.values))
    diff_fact = melt_df.rename({0:'pair',1:'diff in '+colname},axis=1)
    diff_fact['gvkey1'] = diff_fact['pair'].apply(lambda x:x[0])
    diff_fact['gvkey2'] = diff_fact['pair'].apply(lambda x:x[1])
    diff_fact = diff_fact.drop('pair',axis=1)
    merged = pd.merge(sim_df,diff_fact,on=['gvkey1','gvkey2']).sort_values(['gvkey1','gvkey2'])
    return merged

def coint_pavlues(list_tick,year):
    data = yf.download(list_tick, start=str(year)+"-01-01", end=str(year)+"-12-31")
    adj_close_data = data[('Adj Close')].dropna(axis=1,how='all')#.isna().sum()>10
    adj_close_data = adj_close_data[[i for i,j in pd.Series(adj_close_data.isna().sum(axis=0)==2).items() if j==True]]
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    scores = []
    pvalues = []
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            scores.append((keys[i], keys[j],result[0]))
            pvalues.append((keys[i], keys[j],result[1]))
            count+=1
        print(str(count)+' Pairs done..!!')
    return scores,pvalues