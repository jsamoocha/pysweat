import pandas as pd
import math


def lat_long_to_x_y(lat_long_df):
    latitude = pd.Series([point[0] for point in lat_long_df.latlng], index=lat_long_df.index)
    longitude = pd.Series([point[1] for point in lat_long_df.latlng], index=lat_long_df.index)
    center_lat = min(latitude) + (max(latitude) - min(latitude)) / 2
    lat_long_df['x'] = [math.radians(lng) * math.cos(math.radians(center_lat)) for lng in longitude]
    lat_long_df['y'] = [math.radians(lat) for lat in latitude]
    return lat_long_df
