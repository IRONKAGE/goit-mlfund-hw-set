import os

def get_california_housing(target_path):
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    data.frame.to_csv(target_path, index=False)
