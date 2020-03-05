def get_outliers(df, column):
    """
    Gets outliers of given `column` in given `df`.

    :param df: pandas.DataFrame, dataframe.
    :param column: str, dataframe's column to get outliers of.
    :return: pandas.DataFrame, dataframe containing outliers.
    """
    iqr = df[column].quantile(q=0.75) - df[column].quantile(q=0.25)
    upper_bound = df[column].quantile(q=0.75) + 1.5 * iqr
    lower_bound = df[column].quantile(q=0.25) - 1.5 * iqr
    return df[(df[column] > upper_bound) | (df[column] < lower_bound)]
