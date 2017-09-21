import pandas as pd
import math


def lat_long_to_x_y(lat_long_df, lat_long_colname='latlng'):
    latitude = pd.Series([point[0] for point in lat_long_df[lat_long_colname]], index=lat_long_df.index)
    longitude = pd.Series([point[1] for point in lat_long_df[lat_long_colname]], index=lat_long_df.index)
    center_lat = min(latitude) + (max(latitude) - min(latitude)) / 2
    return lat_long_df.assign(**{
        'x': [math.radians(lng) * math.cos(math.radians(center_lat)) for lng in longitude],
        'y': [math.radians(lat) for lat in latitude]
    })
