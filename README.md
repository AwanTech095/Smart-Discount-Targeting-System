# Smart Discount Targeting System

A data mining project that segments e-commerce customers and maps them to discount tiers using behavioral features from the Brazilian Olist dataset.

The project builds a customer-level view from raw transactional tables, applies clustering to identify purchasing behavior patterns, assigns discount tiers, and trains a simple predictive model to classify customers into those tiers.

## Project Objective

Uniform discounts can waste revenue on customers who would have purchased anyway, while missing customers who are more price-sensitive. This project uses data mining techniques to support targeted discount decisions by identifying:

- Price-sensitive customers who may need higher discounts
- Moderate customers who may respond to medium discounts
- High-value customers who may need lower discounts or loyalty-focused offers

## Dataset

The project uses Olist e-commerce data, including customers, orders, order items, products, payments, sellers, reviews, geolocation, and product category translations.

Key raw files:

| File | Description |
| --- | --- |
| `olist_customers_dataset.csv` | Customer profile and location data |
| `olist_orders_dataset.csv` | Order status and order timestamps |
| `olist_order_items_dataset.csv` | Product-level order item details |
| `olist_products_dataset.csv` | Product metadata and categories |
| `olist_order_payments_dataset.csv` | Payment method, installments, and values |
| `product_category_name_translation.csv` | Portuguese-to-English category mapping |

Generated analysis files:

| File | Rows | Description |
| --- | ---: | --- |
| `cleaned_transactional_data.csv` | 115,011 | Merged and cleaned transaction-level dataset |
| `customer_level_features.csv` | 93,335 | Aggregated customer behavior features |
| `customer_discount_segments.csv` | 93,335 | Customer features with cluster labels and discount tiers |

## Workflow

1. **Data Loading**
   - Loads Olist customers, orders, order items, products, and payments data.

2. **Data Preprocessing**
   - Merges raw datasets into one transactional dataset.
   - Keeps delivered orders.
   - Converts timestamp fields.
   - Handles missing values.
   - Removes duplicate and incomplete records.

3. **Feature Engineering**
   - Aggregates transactions into customer-level features.
   - Creates behavioral metrics such as total orders, total spend, average order value, purchase frequency, recency, freight behavior, installment usage, promo-month purchases, and cheap-item ratio.

4. **Customer Segmentation**
   - Scales customer features.
   - Uses K-Means clustering.
   - Evaluates candidate cluster counts with elbow and silhouette analysis.
   - Uses three clusters for customer grouping.

5. **Discount Tier Mapping**
   - Maps clusters into business-friendly discount labels:
     - `High Discount`
     - `Medium Discount`
     - `Low Discount`

6. **Visualization**
   - Cluster distribution
   - Spending behavior by cluster
   - Price sensitivity by cluster
   - PCA-based cluster visualization
   - Discount tier distribution
   - Decision tree confusion matrix
   - Feature importance chart

7. **Association Rule Mining**
   - Applies Apriori to product category baskets.
   - Finds that most orders contain a single category, so meaningful multi-item association rules are limited for this dataset.

8. **Predictive Modeling**
   - Trains a Decision Tree classifier to predict discount tiers from customer features.
   - Evaluates predictions with accuracy, classification report, and confusion matrix.

## Repository Contents

| File | Purpose |
| --- | --- |
| `DM_Proj.ipynb` | Main notebook containing preprocessing, feature engineering, clustering, tiering, association mining, modeling, and visualizations |
| `main.py` | Initial script for loading and inspecting core datasets |
| `DM_midsem_content.txt` | Mid-semester project explanation and progress notes |
| `DM_Proj_Proposal.docx` | Project proposal |
| `l232554_DM_Project_FinalReport.docx` | Final report |
| `l232554_MidSem_Data_Mining_Report.docx` | Mid-semester report |
| `*.pptx` | Project presentation files |
| `l232554_DM_Proj_Final.zip` | Archived final project package |

## Requirements

Install Python 3.9+ and the following packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend jupyter
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/AwanTech095/Smart-Discount-Targeting-System.git
cd Smart-Discount-Targeting-System
```

2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend jupyter
```

3. Start Jupyter:

```bash
jupyter notebook
```

4. Open and run:

```text
DM_Proj.ipynb
```

Note: Some notebook cells use Google Colab-style paths such as `/content/olist_customers_dataset.csv`. When running locally, update those paths to the local CSV filenames in the repository root, for example `olist_customers_dataset.csv`.

## Main Outputs

- `cleaned_transactional_data.csv`
- `customer_level_features.csv`
- `customer_discount_segments.csv`

The final output, `customer_discount_segments.csv`, contains each customer's engineered features, cluster assignment, and recommended discount tier.

## Techniques Used

- Data cleaning and preprocessing
- Feature engineering
- Customer-level aggregation
- K-Means clustering
- Elbow method
- Silhouette score
- PCA visualization
- Apriori association rule mining
- Decision Tree classification
- Exploratory data visualization

## Business Value

The system helps convert raw e-commerce transaction data into actionable customer segments. Instead of assigning the same discount to every customer, the business can use customer behavior to decide which customers should receive high, medium, or low discount offers.
