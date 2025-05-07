import pandas as pd

from transformers.drop_columns import DropColumns
from transformers.fill_unknown import FillUnknown
from transformers.date_feature_extractor import DateFeatureExtractor
from transformers.top_n_reducer import TopNReducer
from transformers.top_n_encoder import TopNEncoder
from transformers.target_marker import TargetMarker
from transformers.skewness_log_transformer import SkewnessLogTransformer
from transformers.correlation_filter import CorrelationFilter
from transformers.frequent_label_encoder import FrequentLabelEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder

drop_cols = ['client_id', 'device_model', 'device_screen_resolution']

fillna_cols = ['utm_campaign', 'utm_adcontent', 'utm_keyword',
               'device_os', 'device_brand', 'utm_source']

one_hot_cols = ['device_category', 'device_os', 'device_browser']

utm_top_cols = ['utm_keyword', 'utm_campaign', 'utm_adcontent', 'utm_source', 'utm_medium']

ordinal_cols = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_brand',
                'geo_country', 'geo_city']

numeric_cols = ['year', 'month', 'day', 'dayofweek', 'is_weekend', 'visit_hour']

targets = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
           'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
           'sub_submit_success', 'sub_car_request_submit_click']

numeric_features = ['visit_number', 'utm_source', 'utm_campaign', 'utm_medium',
                    'geo_country', 'visit_hour']


def to_datetime_visit_number(X: pd.DataFrame):
    X = X.copy()
    X['visit_number'] = pd.to_datetime(X['visit_number'])
    return X

def cast_int_columns(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X['visit_number'] = X['visit_number'].astype('int64')
    X['visit_hour'] = X['visit_hour'].astype('int64')
    return X

def replace_rare_geo_country(X: pd.DataFrame, threshold=0.002) -> pd.DataFrame:
    X = X.copy()
    value_counts = X['geo_country'].value_counts(normalize=True)
    frequent = value_counts[value_counts >= threshold].index
    X['geo_country_frequent'] = X['geo_country'].apply(lambda x: x if x in frequent else 'Other')
    return X

rare_geo_pipeline = FunctionTransformer(replace_rare_geo_country, validate=False)

def replace_rare_geo_city(X: pd.DataFrame, threshold=0.01) -> pd.DataFrame:
    X = X.copy()
    value_counts = X['geo_city'].value_counts(normalize=True)
    frequent = value_counts[value_counts >= threshold].index
    X['geo_city_frequent'] = X['geo_city'].apply(lambda x: x if x in frequent else 'Other')
    return X

rare_city_pipeline = FunctionTransformer(replace_rare_geo_city, validate=False)


def build_preprocessing_pipeline(df_hits):
    return Pipeline(steps=[
        ('drop_cols', DropColumns(drop_cols)),
        ('fill_unknown', FillUnknown(fillna_cols)),
        ('date_features', DateFeatureExtractor()),
        ('top3_browser', TopNReducer(column='device_browser', top_n=3)),
        ('top3_os', TopNReducer(column='device_os', top_n=3)),
        ('top100_utm', TopNEncoder(columns=utm_top_cols, top_n=100)),
        ('col_encoding', ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore',
                                         sparse_output=False,
                                         dtype=int),
                 one_hot_cols),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value',
                                           unknown_value=-1),
                 ordinal_cols),
                ('num', 'passthrough', numeric_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform="pandas")),
        ('mark_target', TargetMarker(df_hits, targets)),
        ('visit_number_dt', FunctionTransformer(to_datetime_visit_number, validate=False)),
        ('cast_int64', FunctionTransformer(cast_int_columns, validate=False)),
        ('rare_geo_country', rare_geo_pipeline),
        ('rare_geo_city', rare_city_pipeline),
        ('log_skewed', SkewnessLogTransformer(numeric_cols, threshold=1.0)),
        ('filter_corr', CorrelationFilter(target='y', low_thr=0.005, high_thr=0.85)),
        ('freq_label', FrequentLabelEncoder()),
    ])


