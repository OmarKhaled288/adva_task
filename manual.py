import streamlit as st
import polars as pl
from objects import one_row_schema
from pipeline import preprocess_one_row


def run(model, le):
    st.write("Please enter the following information:")

    # First Row
    col0, col1 = st.columns(2)
    with col0:
        Name = st.text_input("Name")
    with col1:
        SSN = st.text_input("SSN")

    col10, col2, col3 = st.columns(3)
    with col10:
        age = st.number_input("Age")
        num_credit_card = st.number_input("Number of Credit Cards")
        delay_from_due_date = st.number_input("Delay from Due Date")
        outstanding_debt = st.number_input("Outstanding Debt")
        Occupation = st.selectbox("Occupation", [
            'Engineer', 'Architect', 'Manager', 'Media_Manager', 'Scientist',
            'Lawyer', 'Doctor', 'Mechanic', 'Musician', 'Accountant',
            'Teacher', 'Developer', 'Entrepreneur', 'Writer', 'Journalist'
        ])
        
    with col2:
        annual_income = st.number_input("Annual Income")
        interest_rate = st.number_input("Interest Rate")
        num_of_delayed_payment = st.number_input("Number of Delayed Payments")
        credit_utilization_ratio = st.number_input("Credit Utilization Ratio")
        Credit_Mix = st.selectbox("Credit Mix", ['Good', 'Bad', 'Standard'])

    with col3:
        monthly_inhand_salary = st.number_input("Monthly Inhand Salary")
        num_of_loan = st.number_input("Number of Loans")
        changed_credit_limit = st.number_input("Changed Credit Limit")
        total_emi_per_month = st.number_input("Total EMI per Month")
        Payment_of_Min_Amount = st.selectbox("Payment of Min Amount", ['No', 'Yes', 'NM'])

    # Second Row
    col4, col5, col6 = st.columns(3)
    with col4:
        num_bank_accounts = st.number_input("Number of Bank Accounts")
    with col5:
        num_credit_inquiries = st.number_input("Number of Credit Inquiries")
    with col6:
        amount_invested_monthly = st.number_input("Amount Invested Monthly")

    # Third Row
    col7, col8, col9 = st.columns(3)
    with col7:
        monthly_balance = st.number_input("Monthly Balance")
    with col8:
        Month = st.selectbox("Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    with col9:
        Type_of_Loan = st.multiselect("Type of Loan", [
            'AutoLoan', 'Credit-BuilderLoan', 'DebtConsolidationLoan',
            'HomeEquityLoan', 'MortgageLoan', 'NotSpecified',
            'PaydayLoan', 'PersonalLoan', 'StudentLoan'
        ])

    # Fourth Row
    col10, col11 = st.columns(2)
    with col10:
        Payment_Behaviour = st.selectbox("Payment Behaviour", [
            'High_spent_Large_value_payments',
            'Low_spent_Large_value_payments',
            'High_spent_Small_value_payments',
            'High_spent_Medium_value_payments',
            'Low_spent_Small_value_payments',
            'Low_spent_Medium_value_payments'
        ])
    with col11:
        Credit_History_Age = st.text_input("Credit History Age")


    features = {
    "Name": [Name],
    "SSN": [SSN],
    "Age": [age],
    "Annual_Income": [annual_income],
    "Monthly_Inhand_Salary": [monthly_inhand_salary],
    "Num_Bank_Accounts": [num_bank_accounts],
    "Num_Credit_Card": [num_credit_card],
    "Interest_Rate": [interest_rate],
    "Num_of_Loan": [num_of_loan],
    "Delay_from_due_date": [delay_from_due_date],
    "Num_of_Delayed_Payment": [num_of_delayed_payment],
    "Changed_Credit_Limit": [changed_credit_limit],
    "Num_Credit_Inquiries": [num_credit_inquiries],
    "Outstanding_Debt": [outstanding_debt],
    "Credit_Utilization_Ratio": [credit_utilization_ratio],
    "Total_EMI_per_month": [total_emi_per_month],
    "Amount_invested_monthly": [amount_invested_monthly],
    "Monthly_Balance": [monthly_balance],
    "Month": [Month],
    "Occupation": [Occupation],
    "Credit_Mix": [Credit_Mix],
    "Payment_of_Min_Amount": [Payment_of_Min_Amount],
    "Payment_Behaviour": [Payment_Behaviour],
    "Type_of_Loan": [", ".join(Type_of_Loan)],
    "Credit_History_Age": [Credit_History_Age]
}



    if st.button("Predict", key="manual"):
        df = pl.DataFrame(features, schema=one_row_schema)
        
        preprocessed_df = preprocess_one_row(df)
        prediction = model.predict(preprocessed_df)
        st.write(f"Prediction: {le.inverse_transform(prediction)[0]}")
