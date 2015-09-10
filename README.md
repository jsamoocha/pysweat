# pysweat
Endurance sports data transformation and enrichment functions.

Assuming a simple data model consisting of athletes, activities and streams.

Currently divided into 3 packages:

## Persistence
Encapsulating loading and saving logic, currently only supporting MongoDB as storage.

## Transformation
Mapping of one or more attributes of an observation to a new attribute of the same observation. May include
rolling/moving window operations of arbitrary complexity that use the same attributes from other observations.

## Features
Aggregation of observations to a new attribute of a higher-order entity, e.g. every speed measure in a stream
becoming the average speed of an activity.