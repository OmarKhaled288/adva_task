import polars as pl
from objects import num_columns, natural_number_columns, schema, missing_columns

def df_preprocess_cat(df):
    df = (df
          .drop(['Name', 'SSN', 'Num_Credit_Inquiries'])
         )
    df = (df
          .with_columns(pl.when(pl.col('Occupation') != '_______').then(pl.col('Occupation')).otherwise(pl.lit(None)))
         )
    df = (df
          .with_columns(pl.when((pl.col('Age') > 0) & (pl.col('Age') < 100)).then(pl.col('Age')).otherwise(pl.lit(None)))
         )
    df = (df
          .with_columns(pl.when(pl.col("Credit_History_Age") != "NA").then(pl.col("Credit_History_Age").str.extract(r'(\d+)\s+Years', 1).cast(pl.Int64) * 12 + pl.col("Credit_History_Age").str.extract(r'(\d+)\s+Months', 1).cast(pl.Int64)).otherwise(pl.lit(None)))
         )
    df = (df
          .with_columns(pl.when(pl.col('Credit_Mix') != '_').then(pl.col('Credit_Mix')).otherwise(pl.lit(None)))
         )
    df = (df
          .with_columns(pl.when(pl.col('Payment_Behaviour').is_in([
              'High_spent_Small_value_payments',
              'Low_spent_Medium_value_payments',
              'Low_spent_Large_value_payments',
              'High_spent_Medium_value_payments',
              'Low_spent_Small_value_payments',
              'High_spent_Large_value_payments'
              ])).then(pl.col('Payment_Behaviour')).otherwise(pl.lit(None)))
         )
    df = (df
          .with_columns(pl.col("Type_of_Loan").str.replace(' and', '').replace(' ', '').str.split(', '))
         )

    return df

def zero_check(df):
    df = df.with_columns([pl.when(pl.col(col) > 0).then(pl.col(col)).otherwise(None).alias(col) for col in natural_number_columns])
    return df

def check_cv(df, col):
    cv = df[col].std() / abs(df[col].mean())
    if cv >= 0.5:
        return True
    return False

def calculate_bounds(df, col, iqr_k):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr
    
    return lower, upper

def remove_outliers(df, col, q1, q3):
    df = df.with_columns(
        pl.when((pl.col(col) < q1) | (pl.col(col) > q3))
            .then(pl.lit(None))
            .otherwise(pl.col(col))
            .alias(col))
    return df

def fill_null_nums(df, col):
    df = df.with_columns(pl.col(col).fill_null(pl.col(col).drop_nulls().mode().first())).with_columns(pl.col(col).fill_null(0))
    return df

def preprocess_num(df):
    for col in num_columns:
        if df[col].is_null().sum() == df[col].shape[0] or df[col].sum() == df[col].mean():
            pass
        elif check_cv(df, col):
            lower, upper = calculate_bounds(df, col, 0.01)
            df = remove_outliers(df, col, lower, upper)
    
        df = fill_null_nums(df, col)

    return df

def encode_type_of_loan(df):    
    exploded = df.select(["Type_of_Loan"]).drop_nulls().explode("Type_of_Loan")
    unique_types = exploded["Type_of_Loan"].unique().to_list()
    binary_exprs = [
            pl.col("Type_of_Loan").list.contains(loan_type).cast(pl.Int8).alias(loan_type)
            for loan_type in unique_types
        ]
    encoded = df.select([
        pl.col("Type_of_Loan"),
        *binary_exprs
    ]).drop("Type_of_Loan").fill_null(0)
    result = pl.concat(
        [df.drop(["Type_of_Loan"]), encoded],
        how="horizontal"
    )

    return result

def fill_null_cat_groups(df):
    df = df.with_columns(pl.col('Occupation').fill_null(pl.col('Occupation').drop_nulls().mode().first()))
    df = df.with_columns(pl.col('Credit_Mix').fill_null(pl.col('Credit_Mix').drop_nulls().mode().first()))
    df = df.with_columns(pl.col('Payment_Behaviour').fill_null(pl.col('Payment_Behaviour').drop_nulls().mode().first()))
    df = df.with_columns(pl.col('Type_of_Loan').fill_null(pl.col('Type_of_Loan').drop_nulls().mode().first()))
    return df

def preprocess_group(group: pl.DataFrame) -> pl.DataFrame:
    group = fill_null_cat_groups(group)
    group = zero_check(group)
    group = preprocess_num(group)
    return group


def preprocess_one_row(df):
    df_processed = df_preprocess_cat(df)
    df_processed = preprocess_group(df_processed)
    df_processed = encode_type_of_loan(df_processed)
    for i in missing_columns:
        if i not in df_processed.columns:
            df_processed = df_processed.with_columns(pl.lit(0).alias(i))
    return df_processed

def preprocess(df):
    df_processed = df_preprocess_cat(df)
    df_processed = df_processed.group_by("Customer_ID").map_groups(preprocess_group, schema=schema).collect()
    df_processed = encode_type_of_loan(df_processed)
    return df_processed.drop(['Customer_ID', 'ID'])
