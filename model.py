# --------------------  IMPORT MODULES AND PACKAGES  ------------------------
import streamlit as st
import altair as alt
import plotly.express as px
from streamlit_card import card
from streamlit_extras import add_vertical_space as avs 
from streamlit_extras.dataframe_explorer import dataframe_explorer 
import inspect

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error
import sqlite3 # for sqlite connectivity




conn = sqlite3.connect('your_database.db')
curser = conn.cursor()


# ------------------- Connecting sql server ----------------------

# --------------------  Creating an dataframe of Sales data -----------------------
sales = "SELECT * FROM sales"
df = pd.read_sql_query(sales,conn)



# -------------------- slicing the df for only needed features -------------------------------

df = df.iloc[:,1:]

# --------------------------- from sales data generating an daily sales quantity of each product -----------------------

# Group by date and item name, then sum the sales
daily_sales = df.groupby(['s_date', 'p_id','p_name'])['quantity'].sum().reset_index()

# -----------------------  sales count for each day sales of each product ---------------------------------

salesCount = df.groupby(['s_date','p_id',])['p_id'].value_counts().rename('Sales_count').reset_index()

#------------------------ merging the sales count and daily sales for more understantability ----------------------------

daily_sales = pd.merge(daily_sales,salesCount,on = ['s_date' , 'p_id'],how='left')

# ----------------- finding is the product has discount at that date --------------

dis = df.groupby(['s_date' , 'p_id'])['p_discount'].sum().reset_index()

# ------------------- if it has set the value of discount to q else 0 --------------------

dis['p_discount'] = dis['p_discount'].apply(lambda x: 1 if x > 0 else 0)

#---------------------- Merge 'dis' with 'df' on 'Date' and 'Item Name'----------------------------

daily_sales = pd.merge(daily_sales, dis, on=['s_date', 'p_id'], how='left')

# -------------------- finding the price of each product at that day -----------------------

price = df.groupby(['s_date','p_id','p_price'])['p_id'].value_counts().reset_index()

# ------------------------------- Drop duplicate rows based on 'Date' and 'Item Name', keeping the first occurrence ------------------------
#  bcz it has many parices on single product at the same day ....  

price = price.drop_duplicates(subset=['s_date', 'p_id'], keep='first')

# ----------------- merging the price with the daily sales ------------------------

daily_sales = pd.merge(daily_sales,price,on=['s_date','p_id'],how='left')

daily_sales = daily_sales.drop(columns=['Sales_count','count'],axis=1)

# --------------------------  extracting featres from the date  -------------------------------------

# Assuming 'daily_sales' DataFrame exists and has a 'Date' column
daily_sales['s_date'] = pd.to_datetime(daily_sales['s_date'])
daily_sales['monthOfyear'] = daily_sales['s_date'].dt.month
daily_sales['DayOfWeek'] = daily_sales['s_date'].dt.dayofweek  # Monday=0, Sunday=6
daily_sales['DayOfMonth'] = daily_sales['s_date'].dt.day
daily_sales['year'] = daily_sales['s_date'].dt.year
daily_sales['Weekend'] = daily_sales['DayOfWeek'].apply(lambda x: 1 if x in [5, 6] else 0)
# Calculate the week of the month
daily_sales['WeekOfMonth'] = (daily_sales['DayOfMonth'] - 1) // 7 + 1
daily_sales['quater'] = daily_sales['s_date'].dt.quarter

# -----------------------------  spliting x and y axis --------------------------

x = daily_sales.drop(columns = ['s_date','quantity','p_name'])
y = daily_sales['quantity']

# ---------------------------  randome forest is the best model for this project ,----------------------------




# -------------------------- predicting future sales --------------------


def future_sales(daily_sales,df,n_days):
    
    
    
    # model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    # Initialize and train the RandomForestRegressor
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    rf_model = RandomForestRegressor()
    rf_model.fit(x, y)   # using the entire dataset to train the model 


    
    product1 = "SELECT * FROM product"
    product = pd.read_sql_query(product1,conn)
    predicted_future_sales = []

    def pfs(date, data):
        weekend = 0
        if pd.to_datetime(date).dayofweek == 5 or pd.to_datetime(date).dayofweek == 6:
            weekend = 1
        # Create a DataFrame instead of a dictionary
        data_df = pd.DataFrame({
            'p_id': [data['p_id']], 
            'p_discount': [data['p_discount']],
            'p_price': [data['p_price']],  # Example Value
            'monthOfyear': [pd.to_datetime(date).month],
            'DayOfWeek': [pd.to_datetime(date).dayofweek],
            'DayOfMonth': [pd.to_datetime(date).day],
            'year': [pd.to_datetime(date).year],
            'Weekend': [weekend],
            'WeekOfMonth': [(pd.to_datetime(date).day - 1) // 7 + 1],
            'quater': [pd.to_datetime(date).quarter]
        })

        predicted_future_sales.append([date, data['p_id'], rf_model.predict(data_df)[0], data['p_discount'], data['p_price']])

    from datetime import timedelta, datetime

    def sales_forcasting(dt_start, dt_end, discounted):
        while dt_start <= dt_end:
            dt_start += timedelta(days=1)
            for _, row in product.iterrows():
                pfs(dt_start, row)

    start_date = datetime.now().date()
    end_date = start_date + timedelta(n_days)

    sales_forcasting(start_date, end_date, 0)


    # Create a DataFrame from the list
    predicted_sales_df = pd.DataFrame(predicted_future_sales)
    predicted_sales_df.columns = ['Date','p_id','Predicted Sales','Discount','Price']

    # ----------------------- total stock need for that specific time line ---------------

    total_stock_needed = predicted_sales_df.groupby(['p_id'])['Predicted Sales'].sum().reset_index()
    total_stock_needed = pd.merge(total_stock_needed, product, on='p_id', how='left')[['p_name', 'Predicted Sales']]

    return total_stock_needed,predicted_sales_df,product

# --------------------------------- about project ----------------------------------





def about_project ():
    st.header('About projuct')
    st.link_button('visit colab file ',url="https://colab.research.google.com/drive/18nnQvtJBtKTy83ZUCSGhfAzXAzpJl7jI?usp=sharing")
    st.text('this is a sales analysis and future sales predition app')
    with st.expander("view source code"):
        st.code(open(__file__).read(), language="python")
    col1,col2 = st.columns((1,1))
    col1.subheader('the sales data')
    col1.write(df)
    product = "SELECT * FROM product"
    product = pd.read_sql_query(product,conn)
    col2.subheader('product data ')
    col2.write(product)
    
    
def product_update():
    st.header('Product update')
    product = "SELECT * FROM product"
    product = pd.read_sql_query(product, conn)
    st.write(product)
    
    st.subheader('Update product')
    
    # Create select options for product name
    p_name = st.selectbox('Select product name', product['p_name'])
    
    # Get the selected product details
    selected_product = product[product['p_name'] == p_name].iloc[0]
    
    # Auto fill the old values
    with st.form(key='update_form'):
        p_price = st.number_input('Product price', value=selected_product['p_price'])
        p_discount = st.number_input('Product discount', value=selected_product['p_discount'])
        submit_button = st.form_submit_button(label='Update product')
    
    if submit_button:
        curser.execute(f"UPDATE product SET p_price = {p_price}, p_discount = {p_discount} WHERE p_name = '{p_name}'")
        conn.commit()
        st.success('Product updated successfully')


def sales_predition():
    with st.sidebar:
        if st.checkbox('enter diff values '):
            days = st.number_input('enter the  : ')
        else:
            days = st.selectbox('select the duration : ',[7,28,29,30])
            
        # pg = st.navigation([
                    
        #   ])

    total_stock_needed,predicted_sales_df,product = future_sales(daily_sales,df,days)
    # Form to update discount and price
    total_stock_needed = total_stock_needed.sort_values(by='Predicted Sales',ignore_index=True)
    col1,col2 = st.columns((1,1))
    col1.subheader('Product Table')
    col1.write(product)
    col2.subheader(f'Daily sales of each product next {days} days')
    avs.add_vertical_space(2) 
    col2.write(predicted_sales_df) 
    
    st.divider()    
    avs.add_vertical_space(2)
    col1.title(f'This gona be the next {days} days sales ')
    avs.add_vertical_space(2) 
    col1,col2 = st.columns((1,2)) 
    avs.add_vertical_space(1) 
    col1.write(total_stock_needed)
    col2.bar_chart(total_stock_needed,x='p_name',y='Predicted Sales',horizontal=True)
    st.divider()
    
    with st.expander("view source code"):
        st.code(inspect.getsource(future_sales), language="python")
    
def sales_analysis():
        filter = st.session_state.get('filter',daily_sales)
        filter = dataframe_explorer(filter)
        
        col2,col3 = st.columns((2,1))
        
        col2.dataframe(filter,use_container_width=True)
        col2.header('Sales in each month')
        col2.bar_chart(filter,x='monthOfyear',y='quantity')

        st.divider()
        col2.header('overall product sales')
        avs.add_vertical_space(3)
        col2.bar_chart(filter,x='p_name',y='quantity')
        col2.header('Sales in each day of week')
        day_of_week_sales = filter.groupby('DayOfWeek')['quantity'].sum().reset_index()
        day_of_week_sales['DayOfWeek'] = day_of_week_sales['DayOfWeek'].replace({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        day_of_week_sales['DayOfWeek'] = pd.Categorical(day_of_week_sales['DayOfWeek'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
        day_of_week_sales = day_of_week_sales.sort_values('DayOfWeek').reset_index(drop=True)
        col2.line_chart(day_of_week_sales, x='DayOfWeek', y='quantity')
        


        
        top_three = filter.groupby('p_name')['quantity'].sum().sort_values(ascending=False).head(3).reset_index()
        col3.header('Top three selling product')
        col3.bar_chart(top_three,x='p_name',y='quantity')
        col3.header('Sales in each quater')
        quater_sales = filter.groupby('quater')['quantity'].sum().reset_index()
        fig = px.pie(quater_sales, values='quantity', names='quater', hole=0.5)
        col3.plotly_chart(fig, use_container_width=True)
        col3.header('Sales in each week of month')
        week_sales = filter.groupby('WeekOfMonth')['quantity'].sum().reset_index()
        fig = px.pie(week_sales, values='quantity', names='WeekOfMonth' )
        col3.plotly_chart(fig, use_container_width=True)
        
        st.header('Sales in each day')
        sales_each_day = filter.groupby('DayOfMonth')['quantity'].sum().reset_index()
        fig_line = px.line(sales_each_day, x='DayOfMonth', y='quantity', markers=True)
        fig_bar = px.bar(sales_each_day, x='DayOfMonth', y='quantity')
        st.plotly_chart(fig_line)
        avs.add_vertical_space(3)
        st.plotly_chart(fig_bar)



# ----------------------------------- creating page --------------------------------

home = st.Page(about_project,title='Home',icon='')
predition = st.Page(sales_predition,title='')
analysis = st.Page(sales_analysis,title='Sales Analysis',icon='')
product = st.Page(product_update,title='Product Update',icon='')

# -----------------------------------  streamlit app -----------------------------------

st.set_page_config(
    page_title="Sales Analysis and predition ",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded")



# ------------------------ navigation bar -------------------------------

pg = st.navigation([
    home,
    analysis, 
    predition,
    product
      ])


pg.run()


