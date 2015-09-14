def delta_constant(df, constant, constant_description, measurement='x'):
    df['d_' + measurement + '_' + constant_description] = df[measurement] - constant
    return df
