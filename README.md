# GLOBAL HOUSEHOLD ELECTRIFICATION 
## Introduction 
Global household electrification refers to the proportion of households worldwide with access to electricity, crucial for poverty reduction, economic growth, and improved living standards. While lacking a universally agreed definition, it generally includes reliable power, cooking facilities, and minimum consumption. The International Energy Agency's framework sets evolving benchmarks based on urban/rural settings. Progress has been made, with global electricity access rising from 71% in 1990 to 87% in 2016, largely due to infrastructure and policy advancements. Despite OECD countries nearing universal access, inequality remains, especially in underdeveloped areas where 13% lacked electricity in 2016. Bridging this gap is a shared responsibility for governments, international bodies, and stakeholders in pursuit of sustainable development and inclusivity. (Ritchie et. al, 2022) 

## Problem Statement 
Extensive research has been conducted on household electrification challenges in Sub-Saharan Africa, the variations in access to electricity within OECD (Organization for Economic Cooperation and Development) countries remain an underexplored area of study. Despite being characterized by high levels of economic development, certain segments of populations within OECD countries continue to face inadequate access to electricity, hindering their socio-economic progress and well-being. The current electrification datasets present a significant challenge due to their inconsistency, inaccuracy, and limited availability. These shortcomings have the potential to impede informed and effective decision-making processes.  

## Aim and Objectives 
The primary aim of this project is to develop a global household predictive model with a specific focus solely on OECD countries. While the objectives are:
to incorporate advanced machine learning techniques and additional dataset to refine the model’s accuracy.
to perform some analysis to uncover patterns, correlations, and trends within the dataset to extract some valuable information. 

## Flow Process
The project follows the following flow process:

![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/33fe4e6f-a4e7-4c6d-a4b1-39f663d2d81b)

## Data Gathering & Cleaning
### Data Gathering
To gather the dynamics of household electrification globally, multiple datasets were sourced from reputable organizations.
1. International Energy Agency (IEA) The IEA dataset constitutes a pivotal source, encompassing valuable insights into electricity generation. Specifically, it spans the period from 2016 to 2023, offering a detailed breakdown of the electricity produced by each country.  
2. Data World: The dataset procured from Data World furnishes a broader perspective, covering a substantial timeframe spanning from 1960 to 2017.  
3. WHO: The WHO dataset contributes a distinct dimension by providing information on the distribution of people in rural and urban areas. 

### Data Cleaning
A comprehensive data cleaning process was undertaken, encompassing the cleaning of three datasets: the initial dataset, the second sourced dataset, and the rural population data. Additionally, a new feature depicting urban population feature was derived from the rural dataset. Subsequently, these datasets were merged, first combining the initial dataset with the second sourced data, and then incorporating the population data. In the final merged dataset, the dataset was used to compute electric rural and electric urban rates, crucial indicators of household electrification progress. This `data cleaning laid the foundation for the project.
To enhance data quality and streamline analysis, certain adjustments were made to the datasets. Notably, data for the year 2023 was excluded from the IEA dataset due to its incomplete nature. Furthermore, a selective approach was taken to curate features, retaining only those relevant to OECD countries.
To handle missing data, the KNNImputer method was used, which imputes missing values by considering neighboring data points. Additionally, to manage the impact of zero values on predictive accuracy, instances with multiple zero values were converted to 'NaN' to indicate missing values. The KNNImputer technique was then applied to these 'NaN' values as well, extending its usage for both missing values and the newly designated 'NaN' entries resulting from zero replacements.

### Exploratory Data Analysis 
EDA is an important step in the data analysis process because it allows us to understand the data before applying any statistical models or making any decisions. The following observations were made: 
1. The geographical distribution of average total electricity values across countries was effectively visualized through a map. This visualization (Figure 1) highlighted the United States (USA) as possessing the highest values among the OECD countries studied.
2. One of the key trends that emerged was revealed by a line chart (Figure 2) showcasing the variation in total electric values over time. Notably, the year 2014 stood out as the period with the highest electric value, indicating a potential significant event or trend that impacted electricity consumption during that year.
3. Delving deeper into the distribution of Total Electricity Value, a bar chart (Figure 3) illustrated a skewed pattern. This asymmetry in distribution has implications for understanding the disparities in electricity consumption across OECD countries.
4. Examining rural populations in various countries, a bar chart (Figure 4) identified Slovenia, Portugal, and Slovakia as the top three nations with the highest rural population figures.
5. Shifting focus to urban populations, another bar chart (Figure 5) pinpointed Belgium, Iceland, and Israel as countries with the highest urban populations.
6. A separate analysis of electric values across countries, depicted in a bar chart (Figure 6), highlighted the United States (USA), Japan, and France as leading in terms of high electric values.
7. Finally, a heatmap visualization (Figure 7) revealed strong correlations between the 'Value' variable and both 'electric_rural' and 'electric_urban'. Moreover, the high degree of correlation between 'electric_rural' and 'electric_urban' suggests a potential interdependence between rural and urban electricity consumption patterns.

![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/86e32f69-6ff9-4143-9e81-52dcc7784ceb)

Figure 1: Geographical distribution of average total electricity values across countries


![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/366dee3f-0bd2-4505-bffd-2ba485b1099e)

Figure 2: Trend of electric values over time


![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/e474e87b-f7ba-4559-ae83-9bc3eb125ad0)

Figure 3: distribution of Total Electricity Value


![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/09908553-ae14-4796-9760-085e30caca0f)

Figure 4: Countries with the highest rural population


![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/7d2450c9-a33b-4d28-b793-519ea1e989e9)

Figure 5: Countries with the highest urban population


![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/62eedc65-12eb-4f14-a24d-ed307ab21487)

Figure 6: Countries with the highest Electric Values


![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/cf39a577-1429-46db-8e83-2c8f26600af1)

Figure 7: Correlation between the features


## Data Preprocessing 
The data preprocessing phase involves several essential steps to ensure the quality and suitability of the dataset for analysis and modeling. The following procedures were executed in this regard:
1. The training and testing sets were formed from an 80/20 split (respectively) of the dataset.
2. Categorization of 'Value' Variable: The 'Value' variable was transformed from a continuous variable to a categorical one with three classes - 'Low', 'Medium', and 'High', using the qcut method. This categorization facilitates a more intuitive interpretation of electricity consumption levels.
3. Class Distribution Analysis: The class distribution of the newly categorized 'Value' variable was examined. The number of unique classes, their labels, and their proportional representation within the dataset were calculated and visualized using a bar plot (Figure 8). It was discovered that the ‘value’ variable which is the target variable is unbalanced.
4. Train-Test Split: The dataset was split into training and testing sets using the train-test split function from the sklearn library. Features were separated from the target variable to facilitate model training and evaluation in a ratio of 80% for training data and 20% for testing data, respectively.
5. Label Encoding: Categorical columns such as 'Location' were encoded using the LabelEncoder to convert their categorical values into numerical values. This encoding step is necessary for most machine learning algorithms. Both the training and testing sets were encoded accordingly.
6. Data Balancing: The training set was balanced using the Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance issues. SMOTE generates synthetic samples of the minority class to match the distribution of the majority class.
7. Standard Scaling: Numerical features were standardized using the StandardScaler, which scales features to have zero mean and unit variance. This preprocessing step ensures that features are on a similar scale, preventing dominance by features with larger magnitudes

![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/b76693b2-8636-4aca-aabb-65cd8843577a)

Figure 8: Imbalanced Target variable (Value)

## Model Development
During the model development phase, a series of steps were taken to build and evaluate predictive models for the given dataset. Firstly, the baseline accuracy was determined by calculating the normalized maximum value count, resulting in a baseline accuracy of 0.5638. Various machine learning algorithms were then compared for their performance. The models considered included DecisionTreeClassifier (DTC), RandomForestClassifier (RFC), GradientBoostingClassifier (GBC), AdaBoostClassifier (ADBC), XGBClassifier (XGB), LGBMClassifier (LGB), CatBoostClassifier (CBC), Support Vector Classifier (SVC), Gaussian Naive Bayes (GNB), Logistic Regression (LGR), and KNeighborsClassifier (KNC).
Through cross-validation with a 5-fold split, the accuracy scores for each model were computed. The resulting scores revealed the strengths and weaknesses of different algorithms. Additionally, a boxplot (Figure 9) visualization was employed to facilitate a comprehensive comparison of these models based on their accuracy scores.

![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/8d98e86f-33ed-4635-9f62-155dec77b7c4)
![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/21c21094-bed6-4744-930c-7dfd19b38ab9)

Figure 9: Comparison of the accuracy results of the models

Further, a specific model, LGBMClassifier, was chosen for further analysis due to its promising performance. This model was fitted to the preprocessed and scaled training data. The training and test accuracies were subsequently assessed, yielding a training accuracy of 1.0 and a test accuracy of 0.9942.
To evaluate the model's performance in more detail, a confusion matrix (Figure 11) was generated, depicting the model's classification results. Additionally, a classification report was produced, offering insights into precision, recall, and F1-score for each class (Figure 10). 
Finally, feature importances were extracted from the LGBMClassifier model and visualized in a bar chart (Figure 12) to identify the significance of different features in predicting the target variable.

![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/b7cbea5e-4d41-4b17-a746-43653aeec81f)

Figure 10: Classification Report


![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/77ff200f-98b5-4d0b-98c4-d4e8587ee299)

Figure 11 : Showing the Confusion matrix


![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/9e3dc408-0f38-4f5b-9ec8-0fa8edbababe)

Figure 12: Feature Importance

## Model Deployment
The deployment of the developed machine learning model was achieved using Streamlit, a platform for interactive data applications. This involved saving the trained LGBMClassifier model in a Streamlit-compatible format (pickle file), creating a Python script to construct the application's layout and functionality, and storing the code in a GitHub repository for version control. By connecting the GitHub repository to a Streamlit account, the application was deployed, allowing users to interact with the model through a web browser.
Link to the web app: https://team-gcp-hamoye-hdsc-spring-23-capstone-project-nb6paq89gdxucg.streamlit.app/

![image](https://github.com/krissemmy/Team-GCP-Hamoye-HDSC-Spring-23-Capstone-Project/assets/119800888/76f011c9-b87f-4359-bed0-aab9608cc380)

Figure 13: The deployed model web app

## Conclusion 
This research centered on the application of various machine learning algorithms to predict electricity values within OECD countries, thereby contributing to the enhancement of household electricity provisioning. Based on the gathered insights, it is reasonable to conclude that the Light Gradient Boosting Machine outperforms the other machine learning algorithms employed in this investigation for predicting household electrification. 

## Recommendation
Given the context of the previous study which concentrated on Sub-Saharan countries and subsequently this project shifted focus to OECD nations, we suggest a broader approach for the next project, encompassing global perspectives. Specifically, we recommend that the upcoming research explore electricity access and related metrics on a global scale. This collaborative effort would promote a more robust and impactful project across diverse regions and economies.

## REFERENCE
Hannah Ritchie, Max Roser and Pablo Rosado (2022) - "Energy". Published online at OurWorldInData.org. Retrieved from: 'https://ourworldindata.org/energy' [Online Resource]
