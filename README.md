
### Questions

### Objectives
YWBAT 
- apply pacf and acf on our data

### Outline
data found [here](https://data.world/data-society/global-climate-change-data/workspace/file?filename=GlobalLandTemperatures%2FGlobalTemperatures.csv)


```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```


```python
# Load in dataset 
df = pd.read_csv("./data/GlobalLandTemperatures_GlobalTemperatures.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dt</th>
      <th>LandAverageTemperature</th>
      <th>LandAverageTemperatureUncertainty</th>
      <th>LandMaxTemperature</th>
      <th>LandMaxTemperatureUncertainty</th>
      <th>LandMinTemperature</th>
      <th>LandMinTemperatureUncertainty</th>
      <th>LandAndOceanAverageTemperature</th>
      <th>LandAndOceanAverageTemperatureUncertainty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1750-01-01</td>
      <td>3.034</td>
      <td>3.574</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1750-02-01</td>
      <td>3.083</td>
      <td>3.702</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1750-03-01</td>
      <td>5.626</td>
      <td>3.076</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1750-04-01</td>
      <td>8.490</td>
      <td>2.451</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1750-05-01</td>
      <td>11.573</td>
      <td>2.072</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape, df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3192 entries, 0 to 3191
    Data columns (total 9 columns):
    dt                                           3192 non-null object
    LandAverageTemperature                       3180 non-null float64
    LandAverageTemperatureUncertainty            3180 non-null float64
    LandMaxTemperature                           1992 non-null float64
    LandMaxTemperatureUncertainty                1992 non-null float64
    LandMinTemperature                           1992 non-null float64
    LandMinTemperatureUncertainty                1992 non-null float64
    LandAndOceanAverageTemperature               1992 non-null float64
    LandAndOceanAverageTemperatureUncertainty    1992 non-null float64
    dtypes: float64(8), object(1)
    memory usage: 224.5+ KB





    ((3192, 9), None)




```python
### Transform data to time series
df["dt"] = pd.to_datetime(df["dt"])
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3192 entries, 0 to 3191
    Data columns (total 9 columns):
    dt                                           3192 non-null datetime64[ns]
    LandAverageTemperature                       3180 non-null float64
    LandAverageTemperatureUncertainty            3180 non-null float64
    LandMaxTemperature                           1992 non-null float64
    LandMaxTemperatureUncertainty                1992 non-null float64
    LandMinTemperature                           1992 non-null float64
    LandMinTemperatureUncertainty                1992 non-null float64
    LandAndOceanAverageTemperature               1992 non-null float64
    LandAndOceanAverageTemperatureUncertainty    1992 non-null float64
    dtypes: datetime64[ns](1), float64(8)
    memory usage: 224.5 KB



```python
df.set_index(keys=['dt'], inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LandAverageTemperature</th>
      <th>LandAverageTemperatureUncertainty</th>
      <th>LandMaxTemperature</th>
      <th>LandMaxTemperatureUncertainty</th>
      <th>LandMinTemperature</th>
      <th>LandMinTemperatureUncertainty</th>
      <th>LandAndOceanAverageTemperature</th>
      <th>LandAndOceanAverageTemperatureUncertainty</th>
    </tr>
    <tr>
      <th>dt</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1750-01-01</th>
      <td>3.034</td>
      <td>3.574</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1750-02-01</th>
      <td>3.083</td>
      <td>3.702</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1750-03-01</th>
      <td>5.626</td>
      <td>3.076</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1750-04-01</th>
      <td>8.490</td>
      <td>2.451</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1750-05-01</th>
      <td>11.573</td>
      <td>2.072</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 3192 entries, 1750-01-01 to 2015-12-01
    Data columns (total 8 columns):
    LandAverageTemperature                       3180 non-null float64
    LandAverageTemperatureUncertainty            3180 non-null float64
    LandMaxTemperature                           1992 non-null float64
    LandMaxTemperatureUncertainty                1992 non-null float64
    LandMinTemperature                           1992 non-null float64
    LandMinTemperatureUncertainty                1992 non-null float64
    LandAndOceanAverageTemperature               1992 non-null float64
    LandAndOceanAverageTemperatureUncertainty    1992 non-null float64
    dtypes: float64(8)
    memory usage: 224.4 KB


### Data is converted, let's build a plot
---------------------


```python
### make a basic plot of each column with time
df.plot(subplots=True, figsize=(16, 10))
plt.show()
```


![png](lesson-plan-II_files/lesson-plan-II_10_0.png)



```python
### what insights can you find through time series analysis?
# Let's just investiage the past 50 years
df2 = df[df.index > '1969-12-31']
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LandAverageTemperature</th>
      <th>LandAverageTemperatureUncertainty</th>
      <th>LandMaxTemperature</th>
      <th>LandMaxTemperatureUncertainty</th>
      <th>LandMinTemperature</th>
      <th>LandMinTemperatureUncertainty</th>
      <th>LandAndOceanAverageTemperature</th>
      <th>LandAndOceanAverageTemperatureUncertainty</th>
    </tr>
    <tr>
      <th>dt</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1970-01-01</th>
      <td>2.836</td>
      <td>0.084</td>
      <td>8.288</td>
      <td>0.117</td>
      <td>-2.584</td>
      <td>0.099</td>
      <td>13.711</td>
      <td>0.052</td>
    </tr>
    <tr>
      <th>1970-02-01</th>
      <td>3.735</td>
      <td>0.082</td>
      <td>9.543</td>
      <td>0.108</td>
      <td>-2.020</td>
      <td>0.127</td>
      <td>14.022</td>
      <td>0.053</td>
    </tr>
    <tr>
      <th>1970-03-01</th>
      <td>5.272</td>
      <td>0.114</td>
      <td>11.066</td>
      <td>0.180</td>
      <td>-0.545</td>
      <td>0.211</td>
      <td>14.503</td>
      <td>0.058</td>
    </tr>
    <tr>
      <th>1970-04-01</th>
      <td>8.603</td>
      <td>0.066</td>
      <td>14.383</td>
      <td>0.179</td>
      <td>2.739</td>
      <td>0.113</td>
      <td>15.440</td>
      <td>0.051</td>
    </tr>
    <tr>
      <th>1970-05-01</th>
      <td>11.206</td>
      <td>0.099</td>
      <td>17.165</td>
      <td>0.121</td>
      <td>5.402</td>
      <td>0.107</td>
      <td>16.104</td>
      <td>0.055</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Now let's plot it again
df2.plot(subplots=True, figsize=(16, 10))
plt.show()
```


![png](lesson-plan-II_files/lesson-plan-II_12_0.png)


### Okay, let's try and find some kind of correlation here



```python
df2.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LandAverageTemperature</th>
      <th>LandAverageTemperatureUncertainty</th>
      <th>LandMaxTemperature</th>
      <th>LandMaxTemperatureUncertainty</th>
      <th>LandMinTemperature</th>
      <th>LandMinTemperatureUncertainty</th>
      <th>LandAndOceanAverageTemperature</th>
      <th>LandAndOceanAverageTemperatureUncertainty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>552.000000</td>
      <td>552.000000</td>
      <td>552.000000</td>
      <td>552.000000</td>
      <td>552.000000</td>
      <td>552.000000</td>
      <td>552.000000</td>
      <td>552.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.128788</td>
      <td>0.087043</td>
      <td>14.823933</td>
      <td>0.122397</td>
      <td>3.518391</td>
      <td>0.131946</td>
      <td>15.575864</td>
      <td>0.057906</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.151471</td>
      <td>0.027095</td>
      <td>4.263117</td>
      <td>0.045466</td>
      <td>4.032501</td>
      <td>0.051693</td>
      <td>1.235743</td>
      <td>0.005561</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.882000</td>
      <td>0.034000</td>
      <td>7.392000</td>
      <td>0.044000</td>
      <td>-3.549000</td>
      <td>0.045000</td>
      <td>13.298000</td>
      <td>0.042000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.997500</td>
      <td>0.067750</td>
      <td>10.636500</td>
      <td>0.091000</td>
      <td>-0.507500</td>
      <td>0.095000</td>
      <td>14.405000</td>
      <td>0.054000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.347000</td>
      <td>0.084000</td>
      <td>15.127500</td>
      <td>0.113000</td>
      <td>3.678000</td>
      <td>0.125000</td>
      <td>15.590500</td>
      <td>0.058000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.153250</td>
      <td>0.101000</td>
      <td>18.978500</td>
      <td>0.145000</td>
      <td>7.418000</td>
      <td>0.157000</td>
      <td>16.749000</td>
      <td>0.062000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.482000</td>
      <td>0.221000</td>
      <td>21.320000</td>
      <td>0.344000</td>
      <td>9.715000</td>
      <td>0.433000</td>
      <td>17.611000</td>
      <td>0.079000</td>
    </tr>
  </tbody>
</table>
</div>




```python
from statsmodels.tsa.seasonal import seasonal_decompose
```


```python
sd = seasonal_decompose(df2)
```


```python
### is there a trend in temperature over time? 
diff_1 = df2.diff(periods=15)
diff_1['LandAverageTemperature'].plot(figsize=(16, 10), subplots=True, style='r.')
plt.show()
```


![png](lesson-plan-II_files/lesson-plan-II_17_0.png)



```python
df_annual = df.LandAverageTemperature.resample('A')
```


```python
df_annual_mean = df_annual.mean()
df_annual_mean.head()
```




    dt
    1750-12-31    8.719364
    1751-12-31    7.976143
    1752-12-31    5.779833
    1753-12-31    8.388083
    1754-12-31    8.469333
    Freq: A-DEC, Name: LandAverageTemperature, dtype: float64




```python
df_annual_mean.plot(figsize = (22,8), style = 'b.')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x121066a90>




![png](lesson-plan-II_files/lesson-plan-II_20_1.png)



```python
year_matrix.head()
```


```python
def plot_col_heatmap(column):
    year_temps = df[column].groupby(pd.Grouper(freq='A'))
    temp_annual = pd.DataFrame()
    for yr, group in year_temps:
        temp_annual[yr] = group.values.ravel()
    year_matrix = temp_annual.T


    plt.matshow(temp_annual, cmap=plt.cm.Spectral_r, aspect='auto', interpolation=None)
    plt.yticks(ticks=range(len(temp_annual.index)), labels=temp_annual.index)
    plt.xticks(ticks=range(len(temp_annual.columns)), labels=temp_annual.columns, rotation=90)
    plt.show()
```


```python
drop_cols = [col for col in df2.columns if 'Uncertainty' in col]
df2.drop(drop_cols, axis=1, inplace=True)
```


```python
#What do we notice about the heatmap
# summer is more hot than winter
```


```python
for column in df2.columns:
    print(column)
    plot_col_heatmap(column)
    print("\n\n")
```


```python
for column in df.drop(drop_cols, axis=1).columns:
    print(column)
    plot_col_heatmap(column)
    print("\n\n")
```

### Assessment
- Climate change is real
- Learned to utilize the heatmap to spot trends, Spectral_r
- Workflow: putting things together from lessons

### Beginning PACF and ACF

### What is the purpose of PACF and ACF?
- ACF:  
    * Compare the correlation between 2 dates as described by the lag time
    * Telling us how much the later data depends on earlier data


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LandAverageTemperature</th>
      <th>LandAverageTemperatureUncertainty</th>
      <th>LandMaxTemperature</th>
      <th>LandMaxTemperatureUncertainty</th>
      <th>LandMinTemperature</th>
      <th>LandMinTemperatureUncertainty</th>
      <th>LandAndOceanAverageTemperature</th>
      <th>LandAndOceanAverageTemperatureUncertainty</th>
    </tr>
    <tr>
      <th>dt</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1750-01-01</th>
      <td>3.034</td>
      <td>3.574</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1750-02-01</th>
      <td>3.083</td>
      <td>3.702</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1750-03-01</th>
      <td>5.626</td>
      <td>3.076</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1750-04-01</th>
      <td>8.490</td>
      <td>2.451</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1750-05-01</th>
      <td>11.573</td>
      <td>2.072</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### ACF and PACF of DATA


```python
fig, ax = plt.subplots(figsize=(12,5))
plot_acf(df2['LandAndOceanAverageTemperature'],ax=ax, lags=50)
plt.grid()
```


![png](lesson-plan-II_files/lesson-plan-II_32_0.png)



```python
fig, ax = plt.subplots(figsize=(12,5))
plot_pacf(df2['LandAndOceanAverageTemperature'],ax=ax, lags=50)
plt.grid()
```


![png](lesson-plan-II_files/lesson-plan-II_33_0.png)



```python
df2.ix[48, :]
```

    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.





    LandAverageTemperature                        2.261
    LandAverageTemperatureUncertainty             0.096
    LandMaxTemperature                            7.464
    LandMaxTemperatureUncertainty                 0.109
    LandMinTemperature                           -2.894
    LandMinTemperatureUncertainty                 0.096
    LandAndOceanAverageTemperature               13.300
    LandAndOceanAverageTemperatureUncertainty     0.052
    Name: 1974-01-01 00:00:00, dtype: float64



### Let's look at differences now of 1 and 12


```python
df2_diff12 = df2.diff(periods=12)
df2_diff1 = df2.diff(periods=1)
df2.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LandAverageTemperature</th>
      <th>LandAverageTemperatureUncertainty</th>
      <th>LandMaxTemperature</th>
      <th>LandMaxTemperatureUncertainty</th>
      <th>LandMinTemperature</th>
      <th>LandMinTemperatureUncertainty</th>
      <th>LandAndOceanAverageTemperature</th>
      <th>LandAndOceanAverageTemperatureUncertainty</th>
    </tr>
    <tr>
      <th>dt</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1970-01-01</th>
      <td>2.836</td>
      <td>0.084</td>
      <td>8.288</td>
      <td>0.117</td>
      <td>-2.584</td>
      <td>0.099</td>
      <td>13.711</td>
      <td>0.052</td>
    </tr>
    <tr>
      <th>1970-02-01</th>
      <td>3.735</td>
      <td>0.082</td>
      <td>9.543</td>
      <td>0.108</td>
      <td>-2.020</td>
      <td>0.127</td>
      <td>14.022</td>
      <td>0.053</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(12,5))
plot_acf(df2_diff1['LandAndOceanAverageTemperature'].dropna(),ax=ax, lags=50)
plt.grid()
```


![png](lesson-plan-II_files/lesson-plan-II_37_0.png)



```python
fig, ax = plt.subplots(figsize=(12, 5))
plot_pacf(df_diff1['LandAndOceanAverageTemperature'].dropna(),ax=ax, lags=100);
plt.grid()
```


![png](lesson-plan-II_files/lesson-plan-II_38_0.png)


$$ ARMA = F_1(y_{t-i}) + F_2(\epsilon_i) $$


```python
fig, ax = plt.subplots(figsize=(12,5))
plot_acf(df_diff12['LandAndOceanAverageTemperature'].dropna(),ax=ax, lags=50)
plt.grid()
```


![png](lesson-plan-II_files/lesson-plan-II_40_0.png)



```python
fig, ax = plt.subplots(figsize=(12, 5))
plot_pacf(df_diff12['LandAndOceanAverageTemperature'].dropna(),ax=ax, lags=50)
plt.grid()
```


![png](lesson-plan-II_files/lesson-plan-II_41_0.png)


### Let's look at diffs of ...


```python
for i in range(30, 100):
    print(i)
    df2_diff = df2.diff(periods=i)


    fig, ax = plt.subplots(figsize=(12,5))
    plot_acf(df2_diff['LandAndOceanAverageTemperature'].dropna(),ax=ax, lags=50)
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_pacf(df2_diff['LandAndOceanAverageTemperature'].dropna(),ax=ax, lags=50)
    plt.grid()
    plt.show()
```

    30



![png](lesson-plan-II_files/lesson-plan-II_43_1.png)



![png](lesson-plan-II_files/lesson-plan-II_43_2.png)


    31



![png](lesson-plan-II_files/lesson-plan-II_43_4.png)



![png](lesson-plan-II_files/lesson-plan-II_43_5.png)


    32



![png](lesson-plan-II_files/lesson-plan-II_43_7.png)



![png](lesson-plan-II_files/lesson-plan-II_43_8.png)


    33



![png](lesson-plan-II_files/lesson-plan-II_43_10.png)



![png](lesson-plan-II_files/lesson-plan-II_43_11.png)


    34



![png](lesson-plan-II_files/lesson-plan-II_43_13.png)



![png](lesson-plan-II_files/lesson-plan-II_43_14.png)


    35



![png](lesson-plan-II_files/lesson-plan-II_43_16.png)



![png](lesson-plan-II_files/lesson-plan-II_43_17.png)


    36



![png](lesson-plan-II_files/lesson-plan-II_43_19.png)



![png](lesson-plan-II_files/lesson-plan-II_43_20.png)


    37



![png](lesson-plan-II_files/lesson-plan-II_43_22.png)



![png](lesson-plan-II_files/lesson-plan-II_43_23.png)


    38



![png](lesson-plan-II_files/lesson-plan-II_43_25.png)



![png](lesson-plan-II_files/lesson-plan-II_43_26.png)


    39



![png](lesson-plan-II_files/lesson-plan-II_43_28.png)



![png](lesson-plan-II_files/lesson-plan-II_43_29.png)


    40



![png](lesson-plan-II_files/lesson-plan-II_43_31.png)



![png](lesson-plan-II_files/lesson-plan-II_43_32.png)


    41



![png](lesson-plan-II_files/lesson-plan-II_43_34.png)



![png](lesson-plan-II_files/lesson-plan-II_43_35.png)


    42



![png](lesson-plan-II_files/lesson-plan-II_43_37.png)



![png](lesson-plan-II_files/lesson-plan-II_43_38.png)


    43



![png](lesson-plan-II_files/lesson-plan-II_43_40.png)



![png](lesson-plan-II_files/lesson-plan-II_43_41.png)


    44



![png](lesson-plan-II_files/lesson-plan-II_43_43.png)



![png](lesson-plan-II_files/lesson-plan-II_43_44.png)


    45



![png](lesson-plan-II_files/lesson-plan-II_43_46.png)



![png](lesson-plan-II_files/lesson-plan-II_43_47.png)


    46



![png](lesson-plan-II_files/lesson-plan-II_43_49.png)



![png](lesson-plan-II_files/lesson-plan-II_43_50.png)


    47



![png](lesson-plan-II_files/lesson-plan-II_43_52.png)



![png](lesson-plan-II_files/lesson-plan-II_43_53.png)


    48



![png](lesson-plan-II_files/lesson-plan-II_43_55.png)



![png](lesson-plan-II_files/lesson-plan-II_43_56.png)


    49



![png](lesson-plan-II_files/lesson-plan-II_43_58.png)



![png](lesson-plan-II_files/lesson-plan-II_43_59.png)


    50



![png](lesson-plan-II_files/lesson-plan-II_43_61.png)



![png](lesson-plan-II_files/lesson-plan-II_43_62.png)


    51



![png](lesson-plan-II_files/lesson-plan-II_43_64.png)



![png](lesson-plan-II_files/lesson-plan-II_43_65.png)


    52



![png](lesson-plan-II_files/lesson-plan-II_43_67.png)



![png](lesson-plan-II_files/lesson-plan-II_43_68.png)


    53



![png](lesson-plan-II_files/lesson-plan-II_43_70.png)



![png](lesson-plan-II_files/lesson-plan-II_43_71.png)


    54



![png](lesson-plan-II_files/lesson-plan-II_43_73.png)



![png](lesson-plan-II_files/lesson-plan-II_43_74.png)


    55



![png](lesson-plan-II_files/lesson-plan-II_43_76.png)



![png](lesson-plan-II_files/lesson-plan-II_43_77.png)


    56



![png](lesson-plan-II_files/lesson-plan-II_43_79.png)



![png](lesson-plan-II_files/lesson-plan-II_43_80.png)


    57



![png](lesson-plan-II_files/lesson-plan-II_43_82.png)



![png](lesson-plan-II_files/lesson-plan-II_43_83.png)


    58



![png](lesson-plan-II_files/lesson-plan-II_43_85.png)



![png](lesson-plan-II_files/lesson-plan-II_43_86.png)


    59



![png](lesson-plan-II_files/lesson-plan-II_43_88.png)



![png](lesson-plan-II_files/lesson-plan-II_43_89.png)


    60



![png](lesson-plan-II_files/lesson-plan-II_43_91.png)



![png](lesson-plan-II_files/lesson-plan-II_43_92.png)


    61



![png](lesson-plan-II_files/lesson-plan-II_43_94.png)



![png](lesson-plan-II_files/lesson-plan-II_43_95.png)


    62



![png](lesson-plan-II_files/lesson-plan-II_43_97.png)



![png](lesson-plan-II_files/lesson-plan-II_43_98.png)


    63



![png](lesson-plan-II_files/lesson-plan-II_43_100.png)



![png](lesson-plan-II_files/lesson-plan-II_43_101.png)


    64



![png](lesson-plan-II_files/lesson-plan-II_43_103.png)



![png](lesson-plan-II_files/lesson-plan-II_43_104.png)


    65



![png](lesson-plan-II_files/lesson-plan-II_43_106.png)



![png](lesson-plan-II_files/lesson-plan-II_43_107.png)


    66



![png](lesson-plan-II_files/lesson-plan-II_43_109.png)



![png](lesson-plan-II_files/lesson-plan-II_43_110.png)


    67



![png](lesson-plan-II_files/lesson-plan-II_43_112.png)



![png](lesson-plan-II_files/lesson-plan-II_43_113.png)


    68



![png](lesson-plan-II_files/lesson-plan-II_43_115.png)



![png](lesson-plan-II_files/lesson-plan-II_43_116.png)


    69



![png](lesson-plan-II_files/lesson-plan-II_43_118.png)



![png](lesson-plan-II_files/lesson-plan-II_43_119.png)


    70



![png](lesson-plan-II_files/lesson-plan-II_43_121.png)



![png](lesson-plan-II_files/lesson-plan-II_43_122.png)


    71



![png](lesson-plan-II_files/lesson-plan-II_43_124.png)



![png](lesson-plan-II_files/lesson-plan-II_43_125.png)


    72



![png](lesson-plan-II_files/lesson-plan-II_43_127.png)



![png](lesson-plan-II_files/lesson-plan-II_43_128.png)


    73



![png](lesson-plan-II_files/lesson-plan-II_43_130.png)



![png](lesson-plan-II_files/lesson-plan-II_43_131.png)


    74



![png](lesson-plan-II_files/lesson-plan-II_43_133.png)



![png](lesson-plan-II_files/lesson-plan-II_43_134.png)


    75



![png](lesson-plan-II_files/lesson-plan-II_43_136.png)



![png](lesson-plan-II_files/lesson-plan-II_43_137.png)


    76



![png](lesson-plan-II_files/lesson-plan-II_43_139.png)



![png](lesson-plan-II_files/lesson-plan-II_43_140.png)


    77



![png](lesson-plan-II_files/lesson-plan-II_43_142.png)



![png](lesson-plan-II_files/lesson-plan-II_43_143.png)


    78



![png](lesson-plan-II_files/lesson-plan-II_43_145.png)



![png](lesson-plan-II_files/lesson-plan-II_43_146.png)


    79



![png](lesson-plan-II_files/lesson-plan-II_43_148.png)



![png](lesson-plan-II_files/lesson-plan-II_43_149.png)


    80



![png](lesson-plan-II_files/lesson-plan-II_43_151.png)



![png](lesson-plan-II_files/lesson-plan-II_43_152.png)


    81



![png](lesson-plan-II_files/lesson-plan-II_43_154.png)



![png](lesson-plan-II_files/lesson-plan-II_43_155.png)


    82



![png](lesson-plan-II_files/lesson-plan-II_43_157.png)



![png](lesson-plan-II_files/lesson-plan-II_43_158.png)


    83



![png](lesson-plan-II_files/lesson-plan-II_43_160.png)



![png](lesson-plan-II_files/lesson-plan-II_43_161.png)


    84



![png](lesson-plan-II_files/lesson-plan-II_43_163.png)



![png](lesson-plan-II_files/lesson-plan-II_43_164.png)


    85



![png](lesson-plan-II_files/lesson-plan-II_43_166.png)



![png](lesson-plan-II_files/lesson-plan-II_43_167.png)


    86



![png](lesson-plan-II_files/lesson-plan-II_43_169.png)



![png](lesson-plan-II_files/lesson-plan-II_43_170.png)


    87



![png](lesson-plan-II_files/lesson-plan-II_43_172.png)



![png](lesson-plan-II_files/lesson-plan-II_43_173.png)


    88



![png](lesson-plan-II_files/lesson-plan-II_43_175.png)



![png](lesson-plan-II_files/lesson-plan-II_43_176.png)


    89



![png](lesson-plan-II_files/lesson-plan-II_43_178.png)



![png](lesson-plan-II_files/lesson-plan-II_43_179.png)


    90



![png](lesson-plan-II_files/lesson-plan-II_43_181.png)



![png](lesson-plan-II_files/lesson-plan-II_43_182.png)


    91



![png](lesson-plan-II_files/lesson-plan-II_43_184.png)



![png](lesson-plan-II_files/lesson-plan-II_43_185.png)


    92



![png](lesson-plan-II_files/lesson-plan-II_43_187.png)



![png](lesson-plan-II_files/lesson-plan-II_43_188.png)


    93



![png](lesson-plan-II_files/lesson-plan-II_43_190.png)



![png](lesson-plan-II_files/lesson-plan-II_43_191.png)


    94



![png](lesson-plan-II_files/lesson-plan-II_43_193.png)



![png](lesson-plan-II_files/lesson-plan-II_43_194.png)


    95



![png](lesson-plan-II_files/lesson-plan-II_43_196.png)



![png](lesson-plan-II_files/lesson-plan-II_43_197.png)


    96



![png](lesson-plan-II_files/lesson-plan-II_43_199.png)



![png](lesson-plan-II_files/lesson-plan-II_43_200.png)


    97



![png](lesson-plan-II_files/lesson-plan-II_43_202.png)



![png](lesson-plan-II_files/lesson-plan-II_43_203.png)


    98



![png](lesson-plan-II_files/lesson-plan-II_43_205.png)



![png](lesson-plan-II_files/lesson-plan-II_43_206.png)


    99



![png](lesson-plan-II_files/lesson-plan-II_43_208.png)



![png](lesson-plan-II_files/lesson-plan-II_43_209.png)



```python

```
