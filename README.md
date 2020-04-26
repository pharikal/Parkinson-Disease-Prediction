# Parkinson-Disease-Prediction


# ABSTRACT

_This course projects includes analysis on Parkinson&#39;s disease which revolves around two telemonitoring datasets from UCI. The major goal of this course project is to experiment all the aspects covered under CSC 5800 - Intelligent Systems and work towards implementation of those knowledge to develop a near to perfect model in predicting Parkinson&#39;s disease. The more accurate the model are, more chances of artificial systems to predict if the person is having Parkinson&#39;s disease._


# Introduction: Overview

Parkinson&#39;s disease is a neurodegenerative disorder that affects movement.

Approximately 60,000 Americans are diagnosed with Parkinson&#39;s disease each year, and this number does not reflect the thousands of cases that go undetected.More than 10 million people worldwide are living with PD.

[Source:https://www.parkinson.org/Understanding-Parkinsons/Statistics]

It develops over years and the symptoms varies from person to person owing to the diversity of the disease.The cause to this disease remains largely unknown. The usage of speech-based data in the classification of Parkinson disease (PD) has been shown to provide an effect, non-invasive mode of classification. This led to increased interest in speech analysis.

1.
# Data: Dataset Overview

The entire analysis revolves around two datasets obtained from UCI Machine Learning Repository

i) [parkinsons.data](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data) and (will term as Parkinson&#39;s Dataset)

ii) [parkinsons\_updrs.data](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data)(will term as Parkinson&#39;s UPDRS Dataset)

Definitions of the dataset is as follows: [Source:UCI]

[i)](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.names) This dataset consists of a range of biomedical voice measurements from 31 individual, 23 with Parkinson&#39;s disease (PD). Each column in the table is a particular voice measure, and each row corresponds one of 195 voice recording from these individuals. The main aim of the data is to discriminate healthy people from those with PD, according to &quot;status&quot; column which is set to 0 for healthy and 1 for PD.

[ii)](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.names)This dataset is based upon pro-longed experiment which consists of a range of biomedical voice measurements captured from 42 people with early-stage Parkinson&#39;s disease recruited to a six-month trial of a telemonitoring device for remote symptom progression monitoring. The recordings were automatically captured in the patient&#39;s homes.

The reason two datasets are chosen for this analysis will be elaborated later.

  1.
## _Attribute Information:_ _Following are the attribute details about Parkinson&#39;s Data and Parkinson&#39;s UPDRS data._

| **parkinsons.data** |
| --- |
| 1 | MDVP:Fo(Hz) | Average vocal fundamental frequency |
| 2 | MDVP:Fhi(Hz) | Maximum vocal fundamental frequency |
| 3 | MDVP:Flo(Hz) | Minimum vocal fundamental frequency |
| 4 | MDVP:Jitter(%) | Measures of variation in fundamental frequency |
| 5 | MDVP:Jitter(Abs) | Measures of variation in fundamental frequency |
| 6 | MDVP:RAP | Measures of variation in fundamental frequency |
| 7 | MDVP:PPQ | Measures of variation in fundamental frequency |
| 8 | Jitter:DDP | Measures of variation in fundamental frequency |
| 9 | MDVP:Shimmer | Measures of variation in amplitude |
| 10 | MDVP:Shimmer(dB) | Measures of variation in amplitude |
| 11 | Shimmer:APQ3 | Measures of variation in amplitude |
| 12 | Shimmer:APQ5 | Measures of variation in amplitude |
| 13 | MDVP:APQ | Measures of variation in amplitude |
| 14 | Shimmer:DDA | Measures of variation in amplitude |
| 15 | NHR | Measures of ratio of noise to tonal components in the voice |
| 16 | HNR | Measures of ratio of noise to tonal components in the voice |
| 17 | RPDE | Nonlinear dynamical complexity measures |
| 18 | D2 | Nonlinear dynamical complexity measures |
| 19 | DFA | Signal fractal scaling exponent |
| 20 | spread1 | Nonlinear measures of fundamental frequency variation |
| 21 | spread2 | Nonlinear measures of fundamental frequency variation |
| 22 | PPE | Nonlinear measures of fundamental frequency variation |
| 23 | status | Health status of the subject(one)-\&gt; Parkinson&#39;s, (zero)-\&gt; Healthy |

| **parkinsons.updrs.data** |
| --- |
| 1 | subject | Integer that uniquely identifies each subject |
| 2 | age | Integer that depicts age of an individual |
| 3 | sex | gender &#39;0&#39; - male, &#39;1&#39; - female |
| 4 | test\_time | Time since recruitment into the trial. |
| 5 | motor\_updrs | Clinician&#39;s motor UPDRS score |
| 6 | total\_updrs | Clinician&#39;s total UPDRS score, linearly interpolated |
| 7 | Jitter(%) | measure of variation in fundamental frequency |
| 8 | Jitter(Abs) | measure of variation in fundamental frequency |
| 9 | Jitter(RAP) | measure of variation in fundamental frequency |
| 10 | Jitter(PPQ5) | measure of variation in fundamental frequency |
| 11 | Jitter(DDP) | measure of variation in fundamental frequency |
| 12 | Shimmer | measure of variation in amplitude |
| 13 | Shimmer(dB) | measure of variation in amplitude |
| 14 | Shimmer:APQ3 | measure of variation in amplitude |
| 15 | Shimmer:APQ5 | measure of variation in amplitude |
| 16 | Shimmer:APQ11 | measure of variation in amplitude |
| 17 | Shimmer:DDA | measure of variation in amplitude |
| 18 | NHR | measures of ratio of noise to tonal components in the voice |
| 18 | HNR | measures of ratio of noise to tonal components in the voice |
| 20 | RPDE | A nonlinear dynamical complexity measure |
| 21 | DFA |  Signal fractal scaling exponent |
| 22 | PPE |  A nonlinear measure of fundamental frequency variation |

  1.
## _Data Exploratory Analysis_

I plotted both the data on a histogram to understand its distribution.This step is essential fruitful analysis and understand the Parkinson&#39;s disease problem. This kind of analysis is also useful when we do not know what the right range of value is an attribute can hold especially for a dataset like this.

The _parkinsons.data_ has a column called &quot;status&quot; which determines if a person is having Parkinson&#39;s disease.The status of the subject holds two values (one) - Parkinson&#39;s, (zero) – healthy. So, the attributes are plotted based on status which will clearly help us to understand a pattern how factors are contributing to Parkinson&#39;s.

Clearly since I already have the data which has status so I must develop a model based on this data which can predict if a person&#39;s speech symptoms can classify a Parkinson disease problem.



# Conclusion:

-
# Through Random Forest algorithm I came to know age, sex, test-time and DFA are the top-most important factor contributing Parkinson&#39;s disease and I find that is the best algorithm to analyze Parkinson&#39;s disease.
-
# UPDRS dataset is more reliable as it provided better accuracy score and tests are done over a larger span of time and have multiple attributes. So, any algorithm when used have higher reliability over the other dataset.

# Citation

| [1] | A Tsanas, MA Little, PE McSharry, LO Ramig (2009)&#39;Accurate telemonitoring of Parkinson.s disease progression by non-invasive speech tests. |
| --- | --- |
| [2] | A Survey of Machine Learning Based Approaches for Parkinson Disease Prediction. Shubham Bind1 , Arvind Kumar Tiwari2 , Anil Kumar Sahani3 |
| [3] | Feature Relevance Analysis and Classification of Parkinson Disease Tele-Monitoring Data Through Data Mining.Dr.R.Geetha Ramani.G. Sivagami.Shomona Gracia Jacob[[https://pdfs.semanticscholar.org/291e/c451967e81db4f6b2d08090f435990d24a24.pdf](https://pdfs.semanticscholar.org/291e/c451967e81db4f6b2d08090f435990d24a24.pdf)] |
| [4] | Rpubs.com |
| [5] | Conf Proc IEEE Eng Med Biol Soc. 2016 Aug;2016:6389-6392. doi: 10.1109/EMBC.2016.7592190.Support vector machine classification of Parkinson&#39;s disease and essential tremor subjects based on temporal fluctuation.[https://www.ncbi.nlm.nih.gov/pubmed/28269710](https://www.ncbi.nlm.nih.gov/pubmed/28269710) |
| [6] | Random Forest-Based Prediction of Parkinson&#39;s Disease Progression UsingAcoustic, ASR and Intelligibility Features.Alexander Zlotnik, Juan M. Montero, Ruben San-Segundo, Ascension Gallardo-Antol] |
| [7] | [https://machinelearningmastery.com/machine-learning-in-r-step-by-step/](https://machinelearningmastery.com/machine-learning-in-r-step-by-step/) |
| [8] | Wikipedia |
| [9] | Youtube |
