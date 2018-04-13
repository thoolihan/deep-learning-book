import pandas as pd
ESSAY_COUNT = 'essay_count'

def col_name(index):
    return "{}_{}".format(ESSAY_COUNT, index)

def has_content(essay):
    if pd.isnull(essay) or essay == "":
        return 0
    return 1

def add_essays(row):
    return row[col_name(1)] + row[col_name(2)] + row[col_name(3)] + row[col_name(4)]

def count_essays(df):
    df[ESSAY_COUNT] = df.project_essay_1.map(has_content)
    df[ESSAY_COUNT] = df[ESSAY_COUNT].add(df.project_essay_2.map(has_content))
    df[ESSAY_COUNT] = df[ESSAY_COUNT].add(df.project_essay_3.map(has_content))
    df[ESSAY_COUNT] = df[ESSAY_COUNT].add(df.project_essay_4.map(has_content))
    return df

def resources_total(df):
    df['subtotal'] = df['quantity'].multiply(df['price'])
    return pd.DataFrame(df.groupby(['id'])['subtotal'].sum().rename('total'))

