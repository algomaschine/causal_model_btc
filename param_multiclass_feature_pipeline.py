"""
Adaptive Multi-Class XGBoost Pipeline with Dynamic Feature Engineering

This script implements a sophisticated walk-forward backtesting pipeline that
incorporates several advanced concepts to create a robust and adaptive model.

Core Methodology:
1.  Walk-Forward Backtesting: Simulates real-world conditions by iterating
    through time, ensuring the model only uses past data.
2.  Dynamic Time-Series Feature Engineering: Creates "embedding-like" features
    (rolling means, volatility, RSI, etc.) for the target variable on the fly,
    preventing any look-ahead bias.
3.  Adaptive Multi-Class Target: Instead of a simple binary target, it
    classifies the market into 5 states (power_short, short, zero, long,
    power_long). The boundaries for these classes are recalculated monthly
    using an exponentially weighted distribution of past returns to adapt to
    the latest market volatility.
4.  Monthly Feature Selection: Once a month, it re-evaluates all features
    (external and engineered) and selects the most predictive subset for the
    upcoming period.
5.  Expected Value Position Sizing: Uses the full probability distribution
    of the 5 target classes to calculate a nuanced trade score, leading to
    more sophisticated risk allocation.
"""
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import NotFittedError
from bokeh.plotting import figure, gridplot
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
from bokeh.embed import components
from bokeh.resources import CDN
import warnings
import datetime
import pickle
import json
import sys

warnings.filterwarnings('ignore')


# --- Utility Functions for Data Transformation ---

def frac_diff_series(series, d):
    """
    Applies fractional differencing to a pandas Series using an expanding window.
    This is methodologically sound for backtesting as it uses no future info.
    Returns a pandas Series of the same length.
    """
    # 1. Compute weights for the entire series length
    weights = get_weights_ffd(d, len(series))
    
    # 2. Create a new series to store the results
    res = pd.Series(index=series.index, dtype='float64')
    
    # 3. Iterate through the series to compute the differenced values
    for i in range(len(series)):
        # For each point 'i', take the dot product of the historical data
        # up to 'i' and the corresponding weights.
        res.iloc[i] = np.dot(series.iloc[:i+1].values, weights[:i+1][::-1])
        
    return res

def get_weights_ffd(d, n_weights):
    """ Generates weights for fractional differentiation. """
    w = [1.]
    for k in range(1, n_weights):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    return np.array(w)

def create_time_series_features(price_series):
    """
    Creates a DataFrame of engineered features from a price series.
    This must be run on a data window to prevent look-ahead bias.
    """
    features = pd.DataFrame(index=price_series.index)
    
    # Simple lags
    for lag in [1, 2, 3, 5, 10]:
        features[f'price_lag_{lag}'] = price_series.shift(lag)
        
    # Rolling features
    for window in [5, 10, 20]:
        features[f'rolling_mean_{window}d'] = price_series.rolling(window=window).mean()
        features[f'rolling_std_{window}d'] = price_series.rolling(window=window).std()
        features[f'momentum_{window}d'] = price_series.pct_change(periods=window)

    # RSI
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi_14d'] = 100 - (100 / (1 + rs))
    
    return features

def create_rolling_lag_features(feature_df):
    """
    Creates lagged and rolling features for all columns in the given dataframe.
    """
    new_features = pd.DataFrame(index=feature_df.index)
    
    for col in feature_df.columns:
        # Create lags and rolling means for each column
        for lag in [1, 3, 5]:
            new_features[f'{col}_lag_{lag}'] = feature_df[col].shift(lag)
        for window in [5, 10]:
            new_features[f'{col}_rolling_mean_{window}d'] = feature_df[col].rolling(window=window).mean()
            new_features[f'{col}_rolling_std_{window}d'] = feature_df[col].rolling(window=window).std()
            
    return new_features


class WalkForwardMultiClass:
    def __init__(self, feature_filepath, price_filepath, target_pair='AUDCAD'):
        # File paths and configuration
        self.feature_filepath = feature_filepath
        self.price_filepath = price_filepath
        self.target_pair = target_pair
        
        # Backtest parameters
        self.initial_training_days = 150
        self.d_param = 0.5
        self.top_n_features = 30
        
        # Financial parameters
        self.initial_capital = 100000000
        self.base_position_size_pct = 0.1
        
        # State variables
        self.merged_df = None
        self.results = []
        self.current_feature_set = []
        self.class_labels = [0, 1, 2, 3, 4] # power_short, short, zero, long, power_long
        self.class_scores = [-2, -1, 0, 1, 2] # For expected value calculation
        self.output_dir = None # To be defined at runtime
        self.checkpoints_dir = None

    def load_and_prepare_data(self):
        """ Loads, merges, and prepares the initial raw dataset. """
        print("Loading and preparing data...")
        try:
            features = pd.read_csv(self.feature_filepath)
            features['Date'] = pd.to_datetime(features['Date'])
            prices = pd.read_csv(self.price_filepath)
            prices['Date'] = pd.to_datetime(prices['Date'])
        except FileNotFoundError as e:
            print(f"  ERROR: Could not load data file. {e}")
            return False

        target_col = f"{self.target_pair}_close"
        
        # Check for target column in both dataframes to handle cases like BTC
        if target_col in features.columns:
            print(f"  Found target column '{target_col}' in feature file.")
            # Merge other prices into features, avoiding duplicate columns except for the join key 'Date'
            price_cols_to_add = prices.columns.difference(features.columns).tolist() + ['Date']
            self.merged_df = pd.merge(features, prices[price_cols_to_add], on='Date', how='inner').sort_values('Date').reset_index(drop=True)

        elif target_col in prices.columns:
            print(f"  Found target column '{target_col}' in price file.")
            self.merged_df = pd.merge(features, prices[['Date', target_col]], on='Date', how='inner').sort_values('Date').reset_index(drop=True)
        
        else:
            print(f"  ERROR: Target column '{target_col}' not found in either the feature or price file.")
            return False
            
        self.merged_df['target_future_price'] = self.merged_df[target_col].shift(-1)
        self.merged_df.dropna(subset=['target_future_price'], inplace=True)
        
        print(f"  Data loaded. Shape: {self.merged_df.shape}. Date range: {self.merged_df['Date'].min().date()} to {self.merged_df['Date'].max().date()}")
        return True

    def _create_adaptive_targets(self, price_series):
        """ Creates 5 target classes based on EWMA-weighted momentum quantiles. """
        fract_series = frac_diff_series(price_series, d=self.d_param)
        momentum = fract_series.diff()
        weighted_momentum = momentum.ewm(span=22).mean().dropna() # Use EWMA to prioritize recent volatility
        
        # Define adaptive quantile boundaries
        boundaries = weighted_momentum.quantile([0, 0.15, 0.40, 0.60, 0.85, 1.0]).to_numpy()
        boundaries[0] = -np.inf # Ensure the lowest values are included
        boundaries[-1] = np.inf # Ensure the highest values are included
        
        target_classes = pd.cut(weighted_momentum, bins=boundaries, labels=self.class_labels, include_lowest=True)
        return target_classes

    def _prepare_training_data(self, data_window, features_to_use):
        """ Prepares a window of data for training or feature selection. """
        # 1. Create time-series features for the target
        ts_features = create_time_series_features(data_window[f"{self.target_pair}_close"])
        
        # 2. Create rolling/lag features for all external features
        rolling_lag_features = create_rolling_lag_features(data_window[features_to_use])
        
        # 3. Add day of week as a feature
        day_of_week = pd.DataFrame(data_window['Date'].dt.dayofweek, index=data_window.index)
        day_of_week.columns = ['day_of_week']
        
        # 4. Combine all features
        X_raw = pd.concat([data_window[features_to_use], ts_features, rolling_lag_features, day_of_week], axis=1)
        
        # 5. Create adaptive targets
        y_raw = data_window[f"{self.target_pair}_close"]
        y_classes = self._create_adaptive_targets(y_raw)
        
        # 6. Align features and target, dropping NaNs from feature engineering
        aligned_df = pd.concat([X_raw, y_classes.rename('target')], axis=1).dropna()
        
        X_final = aligned_df.drop('target', axis=1)
        y_final = aligned_df['target'].astype(int)
        
        return X_final, y_final

    def run_backtest(self):
        """ Executes the main walk-forward loop. """
        print("\nStarting walk-forward backtest...")
        
        # --- Setup Output Directories ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join('backtest_results', f'{self.target_pair}_{timestamp}')
        self.checkpoints_dir = os.path.join(self.output_dir, 'model_checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        print(f"  Artifacts will be saved to: {self.output_dir}")
        
        model, scaler = None, None
        all_external_features = [col for col in self.merged_df.columns if col not in ['Date', 'target_future_price', f"{self.target_pair}_close"]]

        for i in range(self.initial_training_days, len(self.merged_df)):
            current_date = self.merged_df['Date'].iloc[i-1]
            is_friday = current_date.dayofweek == 4
            is_last_week_of_month = (current_date + pd.Timedelta(days=7)).month != current_date.month

            # --- Monthly Feature Selection & Retraining ---
            if i == self.initial_training_days or (is_friday and is_last_week_of_month):
                print(f"\n--- Monthly Retraining on {current_date.date()} ---")
                train_window = self.merged_df.iloc[:i].copy()
                
                # 1. Feature Selection
                print("  Selecting features...")
                X_fs, y_fs = self._prepare_training_data(train_window, all_external_features)
                if X_fs.empty:
                    print("  Not enough data for feature selection. Skipping.")
                    continue

                # Safeguard: Check if all target classes are present in the training data.
                # If not, the model cannot be trained, so we skip this cycle and use the previous model.
                if len(y_fs.unique()) < len(self.class_labels):
                    print(f"  WARNING: Not all target classes present in training data on {current_date.date()}. Skipping monthly update.")
                    print(f"  Classes present: {np.sort(y_fs.unique())}")
                    continue
                
                fs_model = xgb.XGBClassifier(objective='multi:softprob', num_class=5, n_jobs=1, random_state=42).fit(X_fs, y_fs)
                importances = pd.Series(fs_model.feature_importances_, index=X_fs.columns)
                self.current_feature_set = importances.nlargest(self.top_n_features).index.tolist()
                print(f"  Selected {len(self.current_feature_set)} features. Top 5: {self.current_feature_set[:5]}")

                # 2. Model Training on selected features
                print("  Training main model...")
                
                # We already have the engineered features from the feature selection step.
                # Just select the columns we need from the X_fs dataframe.
                X_train = X_fs[self.current_feature_set]
                y_train = y_fs.loc[X_train.index] # Ensure index alignment

                if X_train.empty:
                    print("  Not enough data for training after feature selection. Skipping.")
                    continue
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                model = xgb.XGBClassifier(objective='multi:softprob', num_class=5, n_jobs=1, random_state=42)
                model.fit(X_train_scaled, y_train)
                print("  Monthly retraining complete.")

                # --- Save artifacts for this training cycle ---
                date_str = current_date.strftime('%Y-%m-%d')
                model_path = os.path.join(self.checkpoints_dir, f'{date_str}_model.json')
                scaler_path = os.path.join(self.checkpoints_dir, f'{date_str}_scaler.pkl')
                features_path = os.path.join(self.checkpoints_dir, f'{date_str}_features.json')
                
                model.save_model(model_path)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                with open(features_path, 'w') as f:
                    json.dump(self.current_feature_set, f)
                print(f"  Saved model, scaler, and feature list to {self.checkpoints_dir}")

            # --- Daily Prediction ---
            if model and scaler and self.current_feature_set:
                prediction_window = self.merged_df.iloc[:i+1].copy() # Data up to and including today
                
                # Engineer all features for the window, then select the ones the model was trained on.
                X_pred_full_eng, _ = self._prepare_training_data(prediction_window, all_external_features)
                if X_pred_full_eng.empty: continue
                
                # Align columns and handle potential missing ones after engineering
                X_pred_selected = X_pred_full_eng[self.current_feature_set]
                
                latest_features = X_pred_selected.iloc[-1:]
                
                try:
                    latest_features_scaled = scaler.transform(latest_features)
                    pred_probas = model.predict_proba(latest_features_scaled)[0]

                    self.results.append({
                        'Date': self.merged_df['Date'].iloc[i],
                        'Actual_Price': self.merged_df['target_future_price'].iloc[i],
                        'Predicted_Probas': pred_probas
                    })
                except NotFittedError:
                    print(f"  Scaler not fitted yet, skipping prediction for {current_date.date()}")
                
        print(f"\n...Backtest loop finished. Generated {len(self.results)} predictions.")
        return True

    def calculate_pnl(self):
        """ Calculates PnL using the expected value of the 5-class output. """
        print("\nCalculating PnL and performance...")
        if not self.results: return False
            
        self.results_df = pd.DataFrame(self.results)
        
        # Calculate expected value score for each prediction
        probas = np.vstack(self.results_df['Predicted_Probas'])
        expected_value_score = np.dot(probas, self.class_scores)
        self.results_df['score'] = expected_value_score
        
        # Position size is proportional to the score
        position_size_pct = self.base_position_size_pct * self.results_df['score']
        
        price_change = self.results_df['Actual_Price'].diff()
        daily_return_pct = (price_change / self.results_df['Actual_Price'].shift(1)) * position_size_pct.shift(1)
        
        self.results_df['daily_pnl'] = daily_return_pct.fillna(0) * self.initial_capital
        self.results_df['cumulative_pnl'] = self.results_df['daily_pnl'].cumsum()
        self.results_df['position_direction_pct'] = position_size_pct * 100
        
        print(f"  PnL calculation complete. Total PnL: ${self.results_df['cumulative_pnl'].iloc[-1]:,.2f}")
        
        # Save full results to CSV
        if self.output_dir:
            results_path = os.path.join(self.output_dir, 'backtest_results.csv')
            self.results_df.to_csv(results_path, index=False)
            print(f"  Full backtest results saved to: {results_path}")

        return True

    def create_report(self):
        """ Generates the final HTML report with Bokeh charts. """
        print("\nGenerating HTML report...")
        if not hasattr(self, 'results_df') or self.results_df.empty: return

        df = self.results_df.rename(columns={'Actual_Price': 'close_price', 'Date': 'date'})
        
        scaler = MinMaxScaler()
        df['close_price_norm'] = scaler.fit_transform(df[['close_price']])
        df['pnl_norm'] = scaler.fit_transform(df[['cumulative_pnl']])
        
        source = ColumnDataSource(df)
        source.data['color'] = ['green' if x > 0 else 'red' for x in df['position_direction_pct']]

        hover = HoverTool(tooltips=[('Date', '@date{%F}'), ('Close', '@close_price{0,0.00}'), ('PnL', '@cumulative_pnl{$0,0}'), ('Position', '@position_direction_pct{0.00}%')], formatters={'@date': 'datetime'}, mode='vline')
        tools = "pan,wheel_zoom,box_zoom,reset,save"

        p1 = figure(height=300, width=900, tools=[tools, hover], x_axis_type='datetime', title=f"{self.target_pair} - Normalized Price vs. PnL")
        p1.line(x='date', y='close_price_norm', source=source, legend_label="Normalized Price", color="blue")
        p1.line(x='date', y='pnl_norm', source=source, legend_label="Normalized PnL", color="green")
        
        p2 = figure(height=200, width=900, x_range=p1.x_range, tools=[tools, hover], x_axis_type='datetime', title="Position Score & Size (%)")
        p2.vbar(x='date', top='position_direction_pct', source=source, width=pd.Timedelta(days=0.5), color='color', alpha=0.7)
        
        chart_layout = gridplot([[p1], [p2]])
        
        script, div = components(chart_layout)
        html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Report for {self.target_pair}</title>{CDN.render()}</head><body><h1>Report for {self.target_pair}</h1>{div}{script}</body></html>"""
        
        filename = os.path.join(self.output_dir, f"multiclass_report_{self.target_pair}.html")
        with open(filename, 'w', encoding='utf-8') as f: f.write(html)
        print(f"  Report saved to: {filename}")


def main():
    """ Main execution function. """
    FEATURE_FILE = 'factors/intermediary_merged_data.csv'
    PRICE_FILE = 'fx/fx_merged_data_fract.csv'
    
    if len(sys.argv) > 1:
        TARGET_PAIR = sys.argv[1]
        print(f"Target pair provided: {TARGET_PAIR}")
    else:
        TARGET_PAIR = 'BTC'
        print(f"No target pair provided, defaulting to: {TARGET_PAIR}")

    pipeline = WalkForwardMultiClass(FEATURE_FILE, PRICE_FILE, TARGET_PAIR)
    
    if pipeline.load_and_prepare_data():
        pipeline.run_backtest()
        pipeline.calculate_pnl()
        pipeline.create_report()
        print("\n[SUCCESS] Backtest completed.")

if __name__ == "__main__":
    main()
