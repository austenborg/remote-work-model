# Trends in Remote Work during the COVID-19 Pandemic

**Austen Borg and Sophia Hoenig**

 **5/7/2022**


## Run Instructions:
The project has four total files, ```data.ipynb``` and ```model.ipynb```, which are Jupyter Notebooks showing our process, and ```logistic_regression.py``` and ```linear_regression.py```, which contain functions used in ```model.ipynb```. 

The ```data.ipynb``` notebook imports 7 raw data files (*.ldjson* format), preprocesses them, and outputs them into 3 different data files to be used in ```model.ipynb``` (*.pkl* format). The preprocessed *.pkl* files will also be included in the data folder below. The data folder should be nested within the docs folder, so that all data files are located at ```./docs/data/filename```. Both notebooks can be run in entirety, with documentation provided throughout. 

## Dependencies
The code has the following dependencies:
```
json == 2.0.9
matplotlib == 3.1.3
numpy == 1.18.1
pandas == 1.3.5
python == 3.7.6
re == 2.2.1
sklearn == 0.22.1
```

The notebooks also use ```os``` and ```datetime``` from the native Python library, as well as custom functions stored in ```logistic_regression.py``` and ```linear_regression.py```. 

## Dataset
The dataset is sourced from Kaggle.com at the following links:

* [May - July 2019](https://www.kaggle.com/datasets/promptcloud/us-job-listings-on-indeed)
* [August - ? 2019](https://www.kaggle.com/datasets/promptcloud/job-listings-on-indeed-usa)
* [July - September 2020](https://www.kaggle.com/datasets/promptcloud/usa-indeed-job-listing)
* [October - December 2021](https://www.kaggle.com/datasets/promptcloud/indeed-usa-job-listing)
* [January - March 2021](https://www.kaggle.com/datasets/promptcloud/indeed-usa-job-listing-dataset)
* [April - June 2021](https://www.kaggle.com/datasets/promptcloud/usa-indeed-job-data)
* [July - September 2021](https://www.kaggle.com/datasets/promptcloud/job-data-usa-indeed)

Due to some formatting issues in the original data, all of the data files can be downloaded in a Google Drive folder ([**link to folder**](https://drive.google.com/drive/folders/1CWVnQrpLDJT2wMtlHXkSUz5tBeQydXNA?usp=sharing)). 
