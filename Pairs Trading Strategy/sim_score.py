import pandas as pd
import os
import sys
#Function to get the similarity scores from year and sic code
def getting_sim(industry_code,year):
  ls = list()
  gvkey2018 = pd.read_csv('Data\\gvkey2018.csv')
  gvkey2018['sic'] = gvkey2018['sic'].astype(str)
  gvkey2018 = gvkey2018[gvkey2018['sic'].str.contains(industry_code,regex=False)]
  #print(gvkey2018)
  gvkey_tic_dict = dict(gvkey2018[['gvkey','tic']].values)
  print("gvkey df prepared..!!")

  sim_df_ls = []
  file_path = "Data\\Similarity Score"
  file_list = os.listdir(file_path)
  for file_name in file_list:
    temp_df = pd.read_csv(os.path.join(file_path,file_name),delimiter="\t",names=['year','gvkey1','gvkey2','score'],engine='python')
    temp_df = temp_df[(temp_df['gvkey1'].isin(gvkey_tic_dict.keys())) & (temp_df['gvkey2'].isin(gvkey_tic_dict.keys()))]
    sim_df_ls.append(temp_df)
    print(file_name+" checked..!!")
  
  sim_df = pd.concat(sim_df_ls)
  sim_df = sim_df.replace({'gvkey1':gvkey_tic_dict,'gvkey2':gvkey_tic_dict})
  sim_df = sim_df.groupby('year').get_group(year)

  return sim_df.to_csv('Saved Sim Matrix\\'+str(year)+'_'+str(industry_code)+'.csv')#sim_df.pivot(index='gvkey2',columns='gvkey1',values='score')

if __name__ == "__main__":
    sic = input("Industry code:")
    year = input("Year:")
    getting_sim(str(sic),year)