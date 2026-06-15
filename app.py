import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Smart Discount Targeting System",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}
.subtitle {
    font-size: 18px;
    color: #9CA3AF;
    margin-bottom: 25px;
}
.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 25px;
}
.insight-box {
    background-color: #111827;
    border: 1px solid #374151;
    padding: 16px;
    border-radius: 12px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛒 Smart Discount Targeting System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Customer segmentation, discount analytics, and behaviour-based discount tier prediction dashboard.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Load Data
# -----------------------------

data = pd.read_csv("customer_discount_segments.csv")
model_data = data.copy()

# -----------------------------
# Behaviour-Based Discount Scoring
# -----------------------------

score_features = [
    "promo_month_order_ratio",
    "high_installment_ratio",
    "avg_installments",
    "recency",
    "frequency",
    "avg_order_value"
]

for col in score_features:
    model_data[col] = model_data[col].fillna(model_data[col].median())

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(model_data[score_features])

scaled_cols = [col + "_scaled" for col in score_features]

scaled_df = pd.DataFrame(
    scaled_values,
    columns=scaled_cols,
    index=model_data.index
)

model_data = pd.concat([model_data, scaled_df], axis=1)

model_data["discount_sensitivity_score"] = (
    0.30 * model_data["promo_month_order_ratio_scaled"] +
    0.25 * model_data["high_installment_ratio_scaled"] +
    0.20 * model_data["avg_installments_scaled"] +
    0.15 * model_data["recency_scaled"] +
    0.10 * (1 - model_data["avg_order_value_scaled"])
)

model_data["discount_tier_behavior"] = pd.qcut(
    model_data["discount_sensitivity_score"],
    q=3,
    labels=["Low Discount", "Medium Discount", "High Discount"]
)

# -----------------------------
# Model Training
# -----------------------------

selected_features = [
    "recency_scaled",
    "high_installment_ratio_scaled",
    "promo_month_order_ratio_scaled",
    "avg_installments_scaled",
    "avg_order_value_scaled"
]

X = model_data[selected_features]
y = model_data["discount_tier_behavior"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# Metrics
# -----------------------------

st.markdown('<div class="section-title"> Project Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", f"{len(model_data):,}")
c2.metric("Customer Clusters", model_data["cluster"].nunique())
c3.metric("Discount Tiers", model_data["discount_tier"].nunique())
c4.metric("Model Accuracy", f"{accuracy:.2%}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------

tab1, tab2, tab3 = st.tabs([
    " Analytics Dashboard",
    " Model Insights",
    " Predict Discount Tier"
])

# -----------------------------
# Tab 1: Analytics
# -----------------------------

with tab1:
    st.markdown("### Customer Analytics Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Discount Tier Distribution")

        tier_counts = model_data["discount_tier"].value_counts()

        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.bar(tier_counts.index.astype(str), tier_counts.values)
        ax1.set_title("Distribution of Discount Tiers")
        ax1.set_xlabel("Discount Tier")
        ax1.set_ylabel("Count")
        plt.xticks(rotation=10)
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        st.markdown("#### Orders vs Spending")

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.scatterplot(
            data=model_data,
            x="total_orders",
            y="total_spent",
            hue="cluster",
            alpha=0.55,
            ax=ax2
        )
        ax2.set_title("Customer Segments: Orders vs Spending")
        ax2.set_xlabel("Total Orders")
        ax2.set_ylabel("Total Spending")
        plt.tight_layout()
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Discount Sensitivity Score Distribution")

        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.hist(model_data["discount_sensitivity_score"], bins=30)
        ax3.set_title("Discount Sensitivity Score Distribution")
        ax3.set_xlabel("Sensitivity Score")
        ax3.set_ylabel("Customers")
        plt.tight_layout()
        st.pyplot(fig3)

    with col4:
        st.markdown("#### Average Spending by Original Discount Tier")

        avg_spending = model_data.groupby("discount_tier")["total_spent"].mean().sort_values()

        fig4, ax4 = plt.subplots(figsize=(5, 3))
        ax4.bar(avg_spending.index.astype(str), avg_spending.values)
        ax4.set_title("Average Spending by Tier")
        ax4.set_xlabel("Discount Tier")
        ax4.set_ylabel("Average Spending")
        plt.xticks(rotation=10)
        plt.tight_layout()
        st.pyplot(fig4)

    st.markdown("#### Cluster Feature Comparison")

    selected_cluster_features = [
        "total_spent",
        "avg_order_value",
        "recency",
        "frequency",
        "avg_installments",
        "high_installment_ratio",
        "promo_month_order_ratio",
        "orders_per_month"
    ]

    selected_cluster_features = [
        col for col in selected_cluster_features if col in model_data.columns
    ]

    cluster_means = model_data.groupby("cluster")[selected_cluster_features].mean()

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    cluster_means.T.plot(kind="bar", ax=ax5)
    ax5.set_title("Cluster Feature Comparison")
    ax5.set_xlabel("Features")
    ax5.set_ylabel("Average Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig5)

    st.info(
        "These visuals summarize customer segments, spending behaviour, discount tier distribution, "
        "and the behavioural score used for predictive modeling."
    )

# -----------------------------
# Tab 2: Model Insights
# -----------------------------

with tab2:
    st.markdown("### Predictive Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Feature Importance")

        importance = pd.Series(
            model.feature_importances_,
            index=selected_features
        ).sort_values(ascending=True)

        fig6, ax6 = plt.subplots(figsize=(5, 3))
        ax6.barh(importance.index, importance.values)
        ax6.set_title("Top Features Influencing Discount Tier")
        ax6.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig6)

    with col2:
        st.markdown("#### Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

        fig7, ax7 = plt.subplots(figsize=(5, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax7
        )
        ax7.set_title("Confusion Matrix")
        ax7.set_xlabel("Predicted")
        ax7.set_ylabel("Actual")
        plt.xticks(rotation=15)
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig7)

    st.markdown("""
    <div class="insight-box">
    <b>Model Insight:</b><br>
    Recency is the strongest predictor, meaning inactive customers are more likely to require discounts for re-engagement.
    Installment behaviour and promotion-month purchasing also strongly influence discount sensitivity.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Tab 3: Prediction
# -----------------------------

with tab3:
    st.markdown("### Predict Discount Tier for a New Customer")
    st.write("Enter customer behaviour values below to estimate the most suitable discount tier.")

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input(
            "Recency: days since last purchase",
            min_value=0,
            value=200
        )

        promo_month_order_ratio = st.slider(
            "Promo Month Order Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.30
        )

        avg_order_value = st.number_input(
            "Average Order Value",
            min_value=0.0,
            value=120.0
        )

    with col2:
        high_installment_ratio = st.slider(
            "High Installment Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.20
        )

        avg_installments = st.number_input(
            "Average Installments",
            min_value=1.0,
            value=2.0
        )

        frequency = st.number_input(
            "Purchase Frequency",
            min_value=1,
            value=1
        )

    input_original = pd.DataFrame([{
        "promo_month_order_ratio": promo_month_order_ratio,
        "high_installment_ratio": high_installment_ratio,
        "avg_installments": avg_installments,
        "recency": recency,
        "frequency": frequency,
        "avg_order_value": avg_order_value
    }])

    input_scaled_values = scaler.transform(input_original[score_features])

    input_scaled_df = pd.DataFrame(
        input_scaled_values,
        columns=scaled_cols
    )

    input_final = input_scaled_df[selected_features]

    if st.button("Predict Discount Tier", use_container_width=True):
        prediction = model.predict(input_final)[0]

        st.success(f"Recommended Discount Tier: {prediction}")

        if prediction == "High Discount":
            st.info(
                "This customer is likely discount-sensitive. "
                "A stronger targeted discount or re-engagement offer may be useful."
            )
        elif prediction == "Medium Discount":
            st.info(
                "This customer may respond to moderate or seasonal discount offers."
            )
        else:
            st.info(
                "This customer is less discount-sensitive. "
                "Loyalty rewards or retention offers may be better than heavy discounts."
            )

        st.write("Input Customer Profile")
        st.dataframe(input_original, use_container_width=True)

st.divider()
st.caption("Smart Discount Targeting System | Data Mining Project")