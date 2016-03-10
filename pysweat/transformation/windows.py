from datetime import timedelta


def subtract_n_minutes(secs, minutes=5, minimum_value=0):
    if secs < minutes * 60:
        return minimum_value
    else:
        return secs - minutes * 60


def select_activity_window(activity_df, date, window_size_days):
    return activity_df[(activity_df.start_date_local <= date) &
                       (activity_df.start_date_local > date - timedelta(days=window_size_days))]