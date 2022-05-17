from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from sklearn.metrics import r2_score
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
import pandas as pd
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np

np.random.seed(7)
mx.random.seed(7)

y = pd.read_csv('Data/air_visit_data.csv.zip')
y = y.pivot(index='visit_date', columns='air_store_id')['visitors']
y = y.fillna(0)
y = pd.DataFrame(y.sum(axis=1))
y = y.reset_index(drop=False)
y.columns = ['date', 'y']

start = pd.Timestamp("01-01-2016", freq="H")

X_reservations = pd.read_csv('Data/air_reserve.csv.zip')
X_reservations['visit_date'] = pd.to_datetime(X_reservations['visit_datetime']).dt.date
X_reservations = pd.DataFrame(X_reservations.groupby('visit_date')
                              ['reserve_visitors'].sum())
X_reservations = X_reservations.reset_index(drop=False)

# Convert to datatime for merging correctly
y.date = pd.to_datetime(y.date)
X_reservations.visit_date = pd.to_datetime(X_reservations.visit_date)

# Merging and filling missing dates with 0
y = y.merge(X_reservations, left_on='date', right_on='visit_date', how='left').fillna(0)

# Preparing and merging holidays data
holidays = pd.read_csv('Data/date_info.csv.zip')
holidays.calendar_date = pd.to_datetime(holidays.calendar_date)
y = y.merge(holidays, left_on='date', right_on='calendar_date',
            how='left').fillna(0)

# Preparing the ListDatasets
train_ds = ListDataset([{
    'target': y.loc[:450, 'y'],
    'start': start,
    'feat_dynamic_real': y.loc[:450, ['reserve_visitors', 'holiday_flg']].values}], freq='H')
test_ds = ListDataset([{
    'target': y['y'],
    'start': start,
    'feat_dynamic_real': y.loc[:, ['reserve_visitors', 'holiday_flg']].values}], freq='H')

# # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
# train_ds = ListDataset([{'target': y.loc[:450, 'y'], 'start': start}], freq='H')
# # test dataset: use the whole dataset, add "target" and "start" fields
# test_ds = ListDataset([{'target': y['y'], 'start': start}], freq='H')

results = []
for learning_rate in [1e-4, 1e-2]:
    for num_layers in [2, 5]:
        for num_cells in [30, 100]:
            estimator = DeepAREstimator(
                prediction_length=28,
                freq='H',
                trainer=Trainer(ctx="cpu",  # remove if on Windows
                                epochs=10,
                                learning_rate=learning_rate,
                                num_batches_per_epoch=100
                                ),
                num_layers=num_layers,
                num_cells=num_cells,
            )
            predictor = estimator.train(train_ds)
            predictions = predictor.predict(test_ds)
            r2 = r2_score(list(predictions)[0].quantile(0.5), list(test_ds)[0]['target'][-28:])
            result = [learning_rate, num_layers, num_cells, r2]
            print(result)
            results.append(result)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)
forecasts = list(forecast_it)
tss = list(ts_it)
ts_entry = tss[0]
forecast_entry = forecasts[0]


def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% predictioninterval" for k in prediction_intervals][::-1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


# plot_prob_forecasts(ts_entry, forecast_entry)
# plt.show()

# print(r2_score(list(test_ds)[0]['target'][-28:], predictions))
# plt.plot(predictions)
# plt.plot(list(test_ds)[0]['target'][-28:])
# plt.legend(['predictions', 'actuals'])
# plt.show()
