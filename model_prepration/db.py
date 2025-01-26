import sqlite3
import csv

# Step 3: Create a cursor object to interact with the database
conn = sqlite3.connect('your_database.db')
curser = conn.cursor()

# Step 4: Create a table called 'product' and 'sales'
curser.execute('''CREATE TABLE IF NOT EXISTS product(p_id INTEGER PRIMARY KEY,p_name TEXT,p_price INTEGER NOT NULL , p_discount INTEGER NOT NULL)''')
curser.execute('''CREATE TABLE IF NOT EXISTS sales(sales_id INTEGER PRIMARY KEY,s_date DATE,quantity FLOAT NOT NULL , p_price FLOAT NOT NULL , p_discount INTEGER NOT NULL, p_name TEXT,p_id INTEGER NOT NULL, FOREIGN KEY (p_id) REFERENCES product(p_id))''')

conn.commit()

product = open('item.csv')
product_content = csv.reader(product)
next(product_content)  # Skip the header
insert_product = "INSERT OR IGNORE INTO product(p_id,p_name,p_price,p_discount) VALUES(?,?,?,?)"
curser.executemany(insert_product, product_content)

sales = open('sales_data.csv')
sales_content = csv.reader(sales)
next(sales_content)  # Skip the header
insert_sales = "INSERT OR IGNORE INTO sales(sales_id , s_date , quantity , p_price , p_discount  , p_name , p_id) VALUES(?,?,?,?,?,?,?)"
curser.executemany(insert_sales, sales_content)

# Committing the changes
conn.commit()
conn.close()