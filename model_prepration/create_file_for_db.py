import random
import pandas as pd
import sqlite3


conn = sqlite3.connect('database.db')
# Step 3: Create a cursor object to interact with the database
curser = conn.cursor()

# df = pd.read_csv('daily_sales.csv')
# print(df)

# df2 = df.groupby(['Item Name'])['Unit Selling Price (RMB/kg)'].mean().agg(lambda x: round(x,2)).reset_index()
# discount_mode = df.groupby(['Item Name'])['Discount (Yes/No)'].agg(lambda x: x.mode()[0]).reset_index()
# df = pd.merge(df2,discount_mode,on='Item Name',how = 'left')
# df.insert(0, 'p_id', [random.randint(100000,999999) for _ in range(len(df))])
# print(df)
# df.to_csv('item.csv',index=False,header=False)
select_product = "SELECT * FROM product"

df = pd.read_csv('sales.csv')
print(df)

df2 = pd.read_sql_query(select_product,conn)
print(df2)

df.insert(0, 'Item Code', df.pop('Item Code'))
df = df.rename(columns={'Item Name':'p_name'})
df = pd.merge(df,df2,on='p_name',how='left')
df = df.drop(columns=['p_price','p_discount','Item Code'])
df.insert(0, 'sales_id', df.apply(lambda row:  f"{row['Date']}{row['p_id']}{random.randint(1000, 99999)}", axis=1).str.replace('-',''))
print(df)
df.to_csv('sales_data.csv',index=False,header=False)
