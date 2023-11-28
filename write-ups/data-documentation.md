# Data Source Documentation
> Identify data source for your research project, and document aspects of this data source. Data provenance and explanation of how the data was or will be gathered should be documented. Ownership of the data, including licensing issues for the dataset and possible outputs should be discussed. Possibilities for bias or censoring in the data should be explored.

## Data Source
One of the data sources I will be using is [here](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
This is a dataset of many images of the ASL alphabet. There are no different columns or aspects, just images, in RGB.
### Data Provenance and Gathering (via creator)
SOURCES
- Personal Images captured with the intent to create a dataset

COLLECTION METHODOLOGY
- The dataset was created by Akash Nagaraj and made publicly accessible.
### APA Citation
Akash Nagaraj. (2018). <i>ASL Alphabet</i> [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/29550
### License
[GPL-2](http://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
### Bias
The only possible bias for this dataset would be the fact that the images are only of one person's hands, this means that people of differing skin colors may not be represented in this dataset, and thus the classifier built from this data may not be as accurate for people of differing ethnicities.

## Data Set Creation
Additionally, I will also be creating my own dataset. The dataset above is only of static ASL, just letters of the alphabet. 

I will create a database of videos of movement-based, whole-word signs). This will require a more complicated model to predict.

I will have different people of differing ethnicities sign different signs to create a dataset to predict these movement-based signs.

I will collect this in front of a green screen to create synthetic data by changing the background.

I will create this dataset as an open-source dataset with either the [GPL-2](http://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html) license or the [CC-0](https://creativecommons.org/public-domain/cc0/) license.
