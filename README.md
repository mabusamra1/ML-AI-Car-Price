# Case Study : What drives the price of a car?

[Link to notebook:]  AIML-Portfolio-Car-Price/prompt_II.ipynb at main · bhaswarey/AIML-Portfolio-Likelihood-Accepting-Coupon](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/prompt_II.ipynb) 



## Context

The goal of this project is to understand what factors make a car more or less expensive. This is the second part in the series of studies, focusing on all the phases covering up to "Modeling" phase in the CRISP-DM-DM process (see Figure 1).

 ![CRISP-DM.png](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/CRISP-DM.png) 

**Figure 1 - CRISP-DM process model**



# 1 Business Understanding

## 1.1 Background

The used car dealership is interested in understanding the key drivers for used car prices.  To aid in the identification of these key drivers, the company would like to data science techniques, leveraging the dataset on used cars that was collected over time, to determine what customers value in used cars. As a result, car inventory will be updated to improve overall sales.



 ![bar_year_price_classic_cars1.png](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/bar_year_price_classic_cars1.png) 

**Figure 2 - Price of class cars by year**



Figure 2 provides a view of the distribution of the price of classic cars older than 1960. These cars will be treated as outliers and will be filtered out from the dataset.







![bar_make_price_classic_cars2.png](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/bar_make_price_classic_cars2.png) 

**Figure 3 - Price of classic cars by make**



Figure 3 - provides a view of the distribution of the price of classic cars by maker, that are older than 1960. The figure highlights the fact that the value of some of the car is dependent make, aged, and their type.







![bar_make_price_expensive_classic_cars3.png](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/bar_make_price_expensive_classic_cars3.png) 



**Figure 4 - Classic cars whose value is more than $1M**



Figure 4 - provides a view of the classic cars whose value is more than $1M. 



In summary, the business problem is as follows:

·    Identify the key features that drive the used car prices.

·    Based on the identified features, realign business processes to improved growth and business outcomes.

Important note: Give that the features of classic cars differ significantly from the rest, the associated data will be handled as outliers and will be removed from the dataset.

 



## 1.2 Business Goals and KPI

The business goal is to:

·    Based on the dataset provided, identify the best model   

·    Using the best model, identify the key features that drive the car prices

The features will help dealership to identify cars that consumers value.



# 2 Data Understanding

This section provides information about the data, its description and is its exploration to make sure it fits the business goals.



## 2.1 **Gathering and Describing Data**

This data comes from kaggle and was modified to contain information on 426K cars. 



![original.png](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/original.png?raw=true)



Here is a sample of the data, which provides following features of the cars:

​		`·    	Price of the car`

​		`·    	Make`

​		`·    	Condition`

​		`·    	Cylinders`

​		`·    	Fuel type`

​		`·    	Mileage`

​		`·    	Title status`

​		`·    	Transmission`

​		`·    	Drive type`

​		`·    	Make`

​		`·    	Body type`

​		`·    	Color`

​		`·    	Year manufactured (age)`



The description of the data is as follows:

```
RangeIndex: 426880 entries, 0 to 426879
Data columns (total 18 columns):
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   id            426880 non-null  int64  
 1   region        426880 non-null  object 
 2   price         426880 non-null  int64  
 3   year          425675 non-null  float64
 4   manufacturer  409234 non-null  object 
 5   model         421603 non-null  object 
 6   condition     252776 non-null  object 
 7   cylinders     249202 non-null  object 
 8   fuel          423867 non-null  object 
 9   odometer      422480 non-null  float64
 10  title_status  418638 non-null  object 
 11  transmission  424324 non-null  object 
 12  VIN           265838 non-null  object 
 13  drive         296313 non-null  object 
 14  size          120519 non-null  object 
 15  type          334022 non-null  object 
 16  paint_color   296677 non-null  object 
 17  state         426880 non-null  object 
dtypes: float64(2), int64(2), object(14)
```



Number of categories per 'object' feature is as follows:

```
  Column Name                # of Categories

______________              _________________

region														404
manufacturer											42
model															29649
condition													6
cylinders													8
fuel															5
title_status											6
transmission											3
VIN																118246
drive															3
size															4
type															13
paint_color												12
state															51
```

## 2.2 Early Data **Exploration** and Data Quality Check

Next step is to check the quality of the data. For example, since many of the column/variable is categorical, the summary of the data is checked for the types in each category. By doing this, the step needed for data cleaning or to be transformed is identified. For example, checking for missing/empty values.

Following is the summary statistics (mean, median, min, max, etc.) of the data:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price</th>
      <th>year</th>
      <th>odometer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.268800e+05</td>
      <td>4.268800e+05</td>
      <td>425675.000000</td>
      <td>4.224800e+05</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.311487e+09</td>
      <td>7.519903e+04</td>
      <td>2011.235191</td>
      <td>9.804333e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.473170e+06</td>
      <td>1.218228e+07</td>
      <td>9.452120</td>
      <td>2.138815e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.207408e+09</td>
      <td>0.000000e+00</td>
      <td>1900.000000</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.308143e+09</td>
      <td>5.900000e+03</td>
      <td>2008.000000</td>
      <td>3.770400e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.312621e+09</td>
      <td>1.395000e+04</td>
      <td>2013.000000</td>
      <td>8.554800e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.315254e+09</td>
      <td>2.648575e+04</td>
      <td>2017.000000</td>
      <td>1.335425e+05</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.317101e+09</td>
      <td>3.736929e+09</td>
      <td>2022.000000</td>
      <td>1.000000e+07</td>
    </tr>
  </tbody>
</table>



Following is the category of each column that is of type object:

```
region          [prescott, fayetteville, florida keys, worcest...
manufacturer    [nan, gmc, chevrolet, toyota, ford, jeep, niss...
model           [nan, sierra 1500 crew cab slt, silverado 1500...
condition       [nan, good, excellent, fair, like new, new, sa...
cylinders       [nan, 8 cylinders, 6 cylinders, 4 cylinders, 5...
fuel                  [nan, gas, other, diesel, hybrid, electric]
title_status    [nan, clean, rebuilt, lien, salvage, missing, ...
transmission                      [nan, other, automatic, manual]
VIN             [nan, 3GTP1VEC4EG551563, 1GCSCSE06AZ123805, 3G...
drive                                        [nan, rwd, 4wd, fwd]
size             [nan, full-size, mid-size, compact, sub-compact]
type            [nan, pickup, truck, other, coupe, SUV, hatchb...
paint_color     [nan, white, blue, red, black, silver, grey, b...
state           [az, ar, fl, ma, nc, ny, or, pa, tx, wa, wi, a...
dtype: object
```



Following are the missing values (in percentage) per column:

```
id               0.00
region           0.00
price            0.00
year             0.28
manufacturer     4.13
model            1.24
condition       40.79
cylinders       41.62
fuel             0.71
odometer         1.03
title_status     1.93
transmission     0.60
VIN             37.73
drive           30.59
size            71.77
type            21.75
paint_color     30.50
state            0.00
dtype: float64
```



Finally, the number of **duplicate rows is 0**.



There is some interesting finding from the summary. For example, there are features like VIN that are not relevant to the price of the car and therefore will be dropped from the dataset. The size has too many missing values and will be dropped. Features like state, model and region have 51, 29649, 404 unique categories subsequently. These features will result in high computation/time cost and therefore can be dropped. Additionally, state and region features are not significant to addressing the business problem at hand. And the manufacturer feature sufficiently compensates for the model feature. Based on the type and significance level (i.e., small percentage of) missing date, the associated rows will either be dropped or filled using SimpleImputer with 'mean' or 'most_frequent' as the strategy.

The data associated with older cars (classic), appeared as outliers because their prices were not driven by features similar to that of non-classic cars. Therefore this data will be extracted from the dataset and analyzed separately.



# 3 Data Preparation

This section provides information on data preparation and cleaning, to allow for analysis as part of this case study and the future case studies covering predictions.  



## 3.1 Data Transformation

Following data attributes that were treated:

**a) year -** 

Converted to 'age' of the car. The missing values were filled using SimpleImputer with 'mean' set as the strategy. Cars older than 1960 were classified as 'classic' (outliers) and were separated out from the dataset.

**b) id, region, state, VIN, size, model -** 

These features were dropped from the data set

**c) odometer -** 

The missing values were filled using SimpleImputer with 'mean' set as the strategy. Rows with mileage above 300K were identified as outliers and were removed.

**d) price -** 

Prices under $1000 and above $75,000 were identified as outliers and therefore the associated data was removed.

e) cylinders -

Give that 0.25% of the rows had cylinders as 'other', they were removed from the data set.



## 3.2 Data Cleansing

In this step, the data is handled based on the problem found during the data understanding phase. Based on the finding, the following steps are executed:

a) The missing values were filled using SimpleImputer with 'mean'/'most_frequent' set as the strategy. 

b) If the percentage of missing values is > 60%, then drop the column.

c) Remove outliers from numerical data.



**Post cleaned data view:**

```
<class 'pandas.core.frame.DataFrame'>
Index: 372616 entries, 0 to 426879
Data columns (total 12 columns):
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   price         372616 non-null  int64  
 1   manufacturer  372616 non-null  object 
 2   condition     372616 non-null  object 
 3   cylinders     372616 non-null  object 
 4   fuel          372616 non-null  object 
 5   odometer      372616 non-null  float64
 6   title_status  372616 non-null  object 
 7   transmission  372616 non-null  object 
 8   drive         372616 non-null  object 
 9   type          372616 non-null  object 
 10  paint_color   372616 non-null  object 
 11  age           372616 non-null  int64  
dtypes: float64(1), int64(2), object(9)
memory usage: 37.0+ MB
```

## 3.3 Final Data

```
        price    manufacturer condition     cylinders      fuel      odometer  \
426875  23590       nissan      good       6 cylinders     gas       32226.0   
426876  30590        volvo      good       6 cylinders     gas       12029.0   
426877  34990     cadillac      good       6 cylinders     diesel    4174.0   
426878  28990        lexus      good       6 cylinders     gas       30112.0   
426879  30590          bmw      good       6 cylinders     gas       22716.0   

          title_status   transmission    drive     type        paint_color  age  
426875        clean        other          fwd      sedan       white         5  
426876        clean        other          fwd      sedan       red           4  
426877        clean        other          4wd      hatchback   white         4  
426878        clean        other          fwd      sedan       silver        6  
426879        clean        other          rwd      coupe       white         5  
```



# 4 Data Understanding - Deep Analysis

This section provides information about deeper exploration and analysis of the data conducted, in preparation for future machine learning model.



## 4.1 Exploratory Data Analysis (EDA)

In this section, the results of exploring and visualizing insight from the data is captured.



### 4.1.1 Price Mileage & Age 

Figure 4 provide the prices of the cars based on its age. The following can be observations from the figure:

a) New cars have higher prices.

b) Cars older than ~50 years begin to go up in price; probably because they become collector/classic cars.







![AIML-Portfolio-Car-Price/images/hist_age_price4.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/hist_age_price4.png) 





**Figure 4 - Price base on the age of the car**



Figure 5 provides the prices of the cars based on the mileage. The following can be observed from the figure:

a) Cars with mileage greater than ~170K begin to diminish in value.

b) In the case of collector cars, mileage may not necessarily impact the price.

c) Cars with low mileage (<~30K) have higher price mostly like because they are newer models.







 ![AIML-Portfolio-Car-Price/images/scatter_odometer_price5.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/scatter_odometer_price5.png) 





**Figure 5 - Price based on the mileage of the car**



Figure 6 provides the car price distribution. The mean is centered some where around ~$15K.



 ![AIML-Portfolio-Car-Price/images/box_hist_price_dist7.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/box_hist_price_dist7.png) 

**Figure 6 - Car price distribution**



Figure 7 provides the car age distribution. The mean is centered somewhere around ~12 years old.



 ![AIML-Portfolio-Car-Price/images/box_hist_age_dist8.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/box_hist_age_dist8.png) 

**Figure 7 - Car age distribution**



Figure 8 provides the car mileage distribution. The mean is centered somewhere around ~90K miles.



 ![AIML-Portfolio-Car-Price/images/box_hist_odometer_dist9.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/box_hist_odometer_dist9.png) 

**Figure 8 - Car mileage distribution**



### 4.1.2 Addressing Business Questions

Following were some of the business questions answered based on the analysis of the data.

**1) How does the car's body type and drive relate to the price?**

Figure 9 provides the car prices based on the car's body type and drive type. 



 ![AIML-Portfolio-Car-Price/images/hist_price_typeDrive10.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/hist_price_typeDrive10.png) 

**Figure 9 - Car prices based in body type and drive type**



The 4wd drive pickup followed by 4wd drive truck have the highest price (~$30K). The other body type cars there were close to the highest value (ranging from ~$27K to ~$21K) were 4wd/rwd drive coupe, convertible, and van subsequently.

The fwd/rwd drive off-road, 4wd/fwd drive mini-van, fwd drive sedan, and fwd drive convertible subsequently had the lowest price. The fwd drive mini-van had higher price then the 4wd drive mini-van and this may be due to other factors like the condition of the mini-van.



**2) How does the car's body type and fuel relate to the price?**

Figure 10 provides the car prices based on the car's body type and the fuel type. 



 ![AIML-Portfolio-Car-Price/images/hist_price_typeFuel11.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/hist_price_typeFuel11.png) 

**Figure 10 - Car prices based in body type and fuel type**



The diesel pickup, diesel truck, and hybrid convertible have the highest prices (> ~$35K). Close followers are electric pickup, electric SUV, electric sedan, diesel sedan, and hybrid pickup (ranging from ~$30K to $33K). The other body/fuel type cars there were close to the highest value (ranging from ~$25K to ~$30K) were gas pickup and diesel van. 

The gas mini-van, were the cheapest (~$10K). Close followers (ranging between ~10K and ~15K) were hybrid hatchback/mini-van, gas hatchback, gas sedan, gas/diesel wagon, and electric/diesel convertible.



**3) What is the make and body type of cars that are at/above $30K?  **

Figure 11 provides the car prices based on the car's body type and the fuel type. 



 ![AIML-Portfolio-Car-Price/images/hist_price_type_make12.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/hist_price_type_make12.png) 

**Figure 11 - Car prices based in make and body type**



Tesla's SUV and wagon along with Aston-martin sedan are the most expensive cars (~$60K). The other cars that are above ~$30K are Aston-martin's convertible and Tesla's sedan/hatchback.



**4) Which are the most expensive classic car?**

Figure 4 proves the most expensive classic cars, which are:

a) Make - Ford, Body Type - sedan, Year-1902, Price-over $1.6M

a) Make - Chevrolet, Body Type - sedan, Year-1955, Price-little over $1.2M



### 4.1.3 Observation  

Initial analysis of the data showed that for non-classic cars, mileage and age are the key performers relative to price, whereas in the case of classic cars, age and make are the key performers. 

The plots show a non-linear relationship between:

a) odometer and price

b) age and price



Figure 12 and 13 provide the maximum and minimum car prices based on individual categorical features. 



 ![AIML-Portfolio-Car-Price/images/catagory_price_mean6.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/catagory_price_max6a.png) 

**Figure 12 - Highest priced cars based on individual categorical features**



 ![AIML-Portfolio-Car-Price/images/catagory_price_mean6.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/catagory_price_min6b.png) 

**Figure 13 - Lowest priced cars based on individual categorical features**



It is clear from the figures that the categorical features have little to no impact when it comes to the price of the cars.  The categorical feature that slightly influence the price is 'manufacturer' (make).

In conclusion, from the EDA conducted, the features that strongly influence the price of the cars are 'odometer' (mileage) and 'age' (year the car was manufactured). 

Before moving on to the modelling phase, we need the make sure the correlation between features is not above 70%. In the case where correlation exceeds 70% then one of the features (assuming two features) must be dropped. 



 ![AIML-Portfolio-Car-Price/images/heatmap_numeric_features13.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/heatmap_numeric_features13.png) 

**Figure 14 - Heatmap of numeric features**



Since the correlation of the features is below 70%, shown in Figure 14, we can keep all of the numeric features.

 

### 5.0   Modeling  

The goal of the modeling phase is to identify the best model to be used for identify the key features that drive the car prices. This phase will have four tasks:

**a) Select modeling technique**: Determine which algorithms to try (e.g. Linear Regression,
Ridge).

**b) Generate test design**: Pending the modeling approach, split the data into training, test, and validation sets.
**c) Build model**: Build and executing the selected models.
**d) Assess model**: Interpret the model results based on the pre-defined success criteria, and the test design.



### 5.1   Model Selection 

The modeling technique is as follows:

### 5.1.1   Pre-process 

The pre-process is defined to transform the categorical features to numerical values,  retaining the original information. As a result, "Ordinal Encoder" was executed against the 'title_status' feature since the categorical values had inherent order to it.

For the rest of the features, "OneHotEncoder" was used because the categorical values did not have inherent order or ranking.

### 5.1.2   Modeling Technique 

The models that will be evaluated are:

-  Linear Regression
- Ridge
- Lasso

The model that performs the best will be selected to identify the key features that drive the car prices. The optimal 'degree' hyperparameter, that has the least MSE/RMSE, will be first identified. The 'GridSearchCV' will be used to identify the 'alpha' hyper parameter. Finally, the optimal model with the lowest MSE/RMSE will be selected. The cross validation parameter will be defaulted to 5, when executing 'GridSearchCV' for Ridge and Lasso.

### 5.1.3   Testing Technique 

The pre-processed data will be split, of which 70% will be randomly allocated for taining and 30% will be allocated for testing.

### 5.1.3   Model Build 

The transformation of the categorical features will be executed as part of the "Pre-process" step. A pipeline will be defined to determine the optimal 'degree' hyperparameter. The numerical features will be transformed as part of this pipeline, defined as follows:

 ![AIML-Portfolio-Car-Price/images/pipeline.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/pipeline.png) 

During the evaluation of the models, the GridSearchCV with CV defaulted to 5 will be executed outside of a pipeline (since we are not setting up the best model to make predictions), to determine the optimal 'alpha' for Redge and Lasso models. The LinearRegression will also be executed outside of the pipeline.

NOTE: If the objective was to make prediction, then a pipeline would have been defined.

### 5.1.4   Model Assessment 

The best performing model with the lowest MSE/RMSE will be selected to identify the features that drive the car price.

### 5.1.5   Summary of Assessment Results 

The dataset was split as follows:

```
X_train shape = (260831, 94)
 X_test shape = (111785, 94)
y_train shape = (260831,)
 y_test shape = (111785,)
```



 ![AIML-Portfolio-Car-Price/images/optimal_degree14.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/optimal_degree14.png) 

**Figure 15 - Optimal 'degree'**

Figure 15 provides the performance of the Linear Regression model in terms of MSE/RMSE as the polynomial 'degree' varies. As shown in the figure, the best 'degree' polynomial is 7.



Following table provides the performance of the Linear Regression, Ridge, and Lasso models.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Model							</th>
      <th>Train MSE					</th>
      <th>Train RMSE				</th>
      <th>Test MSE					</th>
      <th>Test RMSE					</th>
      <th>degree						</th>
      <th>alpha							</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LinearRegression	</th>
      <td>53835950.68468955	</td>
      <td>7337.298595851851	</td>
      <td>54374072.58596747	</td>
      <td>7373.877717047352	</td>
      <td>7									</td>
      <td>N/A								</td>
    </tr>
    <tr>
      <th>Ridge							</th>
      <td>66991603.07616059	</td>
      <td>8184.839832040734	</td>
      <td>67674057.13817555	</td>
      <td>8226.424322764755	</td>
      <td>7									</td>
      <td>122								</td>
    </tr>
    <tr>
      <th>Lasso							</th>
      <td>66992740.730354786</td>
      <td>8184.909329391181	</td>
      <td>67675255.29135343	</td>
      <td>8226.49714589104	</td>
      <td>7									</td>
      <td>3.72759372031494	</td>
    </tr>
  </tbody>
</table>

**Table 1 - Model performance**

From Table 1, LinearRegression model has the best performance and therefore, it was selected to provide the features that drive the price of the car.





 ![AIML-Portfolio-Car-Price/images/best_features15.png at main · bhaswarey/AIML-Portfolio-Car-Price](https://github.com/bhaswarey/AIML-Portfolio-Car-Price/blob/main/images/best_features15.png) 

**Figure 16 - Important features selected by optimal model**

Figure 16 provides the list of features that drive the price of the car. From the figure it is clear that age (year manufactured) of the car has the highest importance followed by mileage (odometer) of the car. The other features of interest that were scored extremely low are:

a) fuel type - diesel

b) fuel type - gas

c) cylinders - 8

d) cylinders - 4

e) body type - pickup

f) body type - sedan

g) body type - truck

h) drive - fwd

i) body type - hatchback



### 6.0   Deployment



### 6.0.1   Summary

Per the results from the analysis of the dataset containing 426K used car, the following were the key findings:

a) The highest important features that had the most impact on the car price were (descending order):

- Year the car was manufactured
- Odometer reading - mileage on the car

b) The next list of features that were identified but had very low importance are

- fuel type - diesel
- fuel type - gas
- cylinders - 8
- cylinders - 4
- body type - pickup
- body type - sedan
- body type - truck
- drive - fwd
- body type - hatchback

c) Cars older than 1960 we classified as "classic"/"collector" cars and their highest important features were

- Year the car was manufactured
- Manufacturer (make) of the car



### 6.0.2   Recommendation

Focus on the following features as they are identified as the most valued by customer in a non-collector car:

- Newer cars (recent manufactured)
- Low mileage

Focus on these features when restocking inventory. Highlight these features, making them standout in used car advertisements. 

In the case of collector car, look for cars older than ~1970 and are either Ford or Chevrolet make. 

The following features can be prioritized over the remaining features of non-collector cars:

- fuel type - diesel
- fuel type - gas
- cylinders - 8
- cylinders - 4
- body type - pickup
- body type - sedan
- body type - truck
- drive - fwd
- body type - hatchback



By aligning inventory and marketing strategies to the recommendations, the dealership will be in a better position to meet its financial goals. Focusing on selling newer cars with low mileage will increase the demand for used cars offered by the dealership. 



### 7.0   Next Steps

The next step would be to cycle through the data preparation to relook at the categorical features with large values by segmenting the dataset to small chunks.  This may help in the reducing the compute and time costs when running it through the models. Example of categorical features to look at are model, city, and state.
