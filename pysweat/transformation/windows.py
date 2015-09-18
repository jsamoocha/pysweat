def subtract_n_minutes(secs, minutes=5, minimum_value=0):
    if secs < minutes * 60:
        return minimum_value
    else:
        return secs - minutes * 60
