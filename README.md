# Sales Forecasting
Sale Forecasting using Sklearn and Keras

## What is Sale Forecasting?
Sales forecasting is the process of predicting a company's future sales revenue over a specific period by analyzing historical data, market trends, customer behavior, and economic conditions.

### Note: Visualization link on 
[Kaggle](https://www.kaggle.com/code/howlingwolfs1989/sales-forecasting)

### About Data 
[Link](https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales)

### Suggest Edits
You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.

#### Files

* train.csv - historical data including Sales
* test.csv - historical data excluding Sales
* sample_submission.csv - a sample submission file in the correct format
* store.csv - supplemental information about the stores
  
#### Data fields

Most of the fields are self-explanatory. The following are descriptions for those that aren't.

* Id - an Id that represents a (Store, Date) duple within the test set

* Store - a unique Id for each store

* Sales - the turnover for any given day (this is what you are predicting)

* Customers - the number of customers on a given day

* Open - an indicator for whether the store was open: 0 = closed, 1 = open

* StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

* SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools

* StoreType - differentiates between 4 different store models: a, b, c, d

* Assortment - describes an assortment level: a = basic, b = extra, c = extended

* CompetitionDistance - distance in meters to the nearest competitor store

* CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened

* Promo - indicates whether a store is running a promo on that day

* Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating

* Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2

* PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

## Data wrangling

### Train

<img width="670" height="198" alt="Capture" src="https://github.com/user-attachments/assets/421bd275-e2fe-4cca-a7be-33396bf0477e" />

### Test

<img width="564" height="213" alt="Capture" src="https://github.com/user-attachments/assets/5367bbed-a57b-45c3-a437-e082df023798" />

### Store

<img width="1121" height="220" alt="Capture" src="https://github.com/user-attachments/assets/1b755b37-7492-48de-b6b6-b4b0e1db03d8" />

### Merging train and store table by outer join. Joining on "Store"

df = pd.merge(train, store, on='Store', how='outer')


<img width="436" height="430" alt="Capture" src="https://github.com/user-attachments/assets/08a4b9be-466e-4932-8b05-5872ba323e64" />


<img width="434" height="127" alt="Capture" src="https://github.com/user-attachments/assets/470d66a1-9a68-41a8-ad7e-f7000a4a127e" />


### Missing Values

From the bar graph we can see missing values:
* CompetitionOpenSince[Month/Year]
* Promo2Since[Year/Week]
* PromoInterval

#### Checking skew
To replace the missing values (impute) with either the <b>mean</b> or <b>median</b> of the column we see how the data is distributed.
* <b>Mean</b> Best when data is <b>normally distributed</b>(no big outliers) - if <b>skewness is close to 0</b>
* <b>Madian</b> Best when data is <b>skewed</b> or has <b>outliers</b> - if <b>skewness is far from 0 (e.g., >1 or <-1)</b>
As you can see the skewness is greater than 1 so we use median

<img width="271" height="364" alt="Missing values" src="https://github.com/user-attachments/assets/c892e760-cd00-4dfa-8164-8b459938a58f" />

<img width="899" height="473" alt="Capture" src="https://github.com/user-attachments/assets/ad0e6f44-4a58-48a1-bb39-42e00a1b0e19" />

### Convert Date column to pandas datetime to extract Year, Months and Days

# Basic Exploration of Data

## Total Sales in years

<img width="158" height="130" alt="Capture" src="https://github.com/user-attachments/assets/c65c4d8b-765c-4ea3-824f-b863fcadf497" />

<img width="1029" height="167" alt="Capture 2" src="https://github.com/user-attachments/assets/8424d4c0-7945-4000-8891-cbb374703f8f" />

## Shopes Open

### Year:

<img width="625" height="437" alt="Capture 3" src="https://github.com/user-attachments/assets/eda3726b-d1ca-4075-97cf-7d9106a4831b" />

### Months:

<img width="1112" height="456" alt="Capture 4" src="https://github.com/user-attachments/assets/10fb0c58-0632-444f-a573-96177f7bd8ba" />

### Days:

<img width="1118" height="439" alt="Capture 5" src="https://github.com/user-attachments/assets/278adb6b-d777-4785-8089-75275ddc5b14" />

### Shope Type:

<img width="1119" height="438" alt="Capture 6" src="https://github.com/user-attachments/assets/da3aabcd-277b-4340-8224-dba05fbdb533" />

## Sales

### Sales
<img width="1115" height="452" alt="Sales" src="https://github.com/user-attachments/assets/a48c9748-2e9d-4bee-8ec4-fe518b7be49d" />

### Sales on Type of Stores
<img width="673" height="311" alt="Most sales on stores" src="https://github.com/user-attachments/assets/84bcb399-3a25-4595-bd3a-f492311a44af" />

### Most Sales in Months
<img width="829" height="312" alt="Most Sales in months" src="https://github.com/user-attachments/assets/553c9860-ade9-46f5-a136-6d7e88f77d93" />

### Sales in Years
<img width="749" height="326" alt="Sales Year" src="https://github.com/user-attachments/assets/0650b2b1-e98a-47f6-b657-12cd69b2947b" />

# Machine Learning
## Sklearn

<img width="403" height="149" alt="Capture" src="https://github.com/user-attachments/assets/dbc5eb01-47f3-4527-b04e-7f1605f06e18" />

## Feature Importance
### RandomForestRegressor

<img width="831" height="470" alt="Capture" src="https://github.com/user-attachments/assets/0e939dd9-bf19-4952-89ad-d205c1eacdc3" />

### XGBRegressor

<img width="830" height="474" alt="Capture 2" src="https://github.com/user-attachments/assets/0af7c9c0-066e-4edc-a473-568b247105a7" />

## Keras

### Architecture

<img width="647" height="492" alt="Keras Model Art" src="https://github.com/user-attachments/assets/ecacd9d4-d265-46ea-8502-88fcd2685804" />

### Loss

<img width="538" height="390" alt="MSE" src="https://github.com/user-attachments/assets/93b52df2-1cb8-4f2c-ba78-3e527f5f375a" />

#### Mean Absolute Error on training set is 302.2367248535156

#### r2 Score =  0.9834996461868286

### Prediction on test as you can see very close to actual values

<img width="207" height="318" alt="Kears Predicts" src="https://github.com/user-attachments/assets/cc14a12a-2684-4bf9-887a-20d92592acc8" />

