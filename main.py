import pandas as pd
import numpy as np

customers = pd.read_csv("olist_customers_dataset.csv")
orders = pd.read_csv("olist_orders_dataset.csv")
order_items = pd.read_csv("olist_order_items_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
payments = pd.read_csv("olist_order_payments_dataset.csv")


#Dataset preprocessing:
print(customers.head())
print(orders.head())
print(order_items.head())
products.head()
payments.head()

'''
print(customers.shape)
print(orders.shape)
print(order_items.shape)
print(products.shape)
print(payments.shape)



customers.columns
orders.columns
order_items.columns
products.columns
payments.columns


customers.isnull().sum()
orders.isnull().sum()
order_items.isnull().sum()
products.isnull().sum()
payments.isnull().sum()

'''