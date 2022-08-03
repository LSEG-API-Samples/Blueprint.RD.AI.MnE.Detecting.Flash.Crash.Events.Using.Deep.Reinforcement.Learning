# Detecting flash crash events using Deep Reinforcement Learning


## <a id="overview"></a>Overview
Deep reinforcement learning (DRL) is one of the most recent Artificial Intelligence (AI) methodologies that has been applied cross-industry in very different use cases. In this Blueprint we will be exploring a DRL AI module that tries to detect the event of a flash crash. Building on the GAN Blueprint Flash Crash synthetic Data with Generative Adversarial Networks we will be using the features generated there, enhance the modelling space a bit more and try to train a DRL agent to understand the environment and give as an early detection signal of some possible abnormality within the market microstructure. We will be using Tensorforce as our Deep Reinforcement Learning framework a very recent module built from the Tensorforce team.

Details and concepts are further explained in the [Detecting flash crash events using Deep Reinforcement Learning](https://developers.refinitiv.com/en/article-catalog/article/modelling-and-evaluation-deep-reinforcement-learning-flash-crashes.html) Blueprint published on the [Refinitiv Developer Community portal](https://developers.refinitiv.com).

## <a id="disclaimer"></a>Disclaimer
The source code presented in this project has been written by Refinitiv only for the purpose of illustrating the concepts of creating example scenarios using the Refinitiv Data Library for Python.

***Note:** To [ask questions](https://community.developers.refinitiv.com/index.html) and benefit from the learning material, I recommend you to register on the [Refinitiv Developer Community](https://developers.refinitiv.com)*

## <a name="prerequisites"></a>Prerequisites

To execute any workbook, refer to the following:

- A Refinitiv Desktop license (Refinitiv Eikon or Refinitiv Workspace) that has API access 
- Tested with Python 3.7.13
- Packages: [pandas](https://pypi.org/project/pandas/), [tensorforce](https://pypi.org/project/Tensorforce/), [refinitiv.data](https://pypi.org/project/refinitiv-data/)
- RD Library for Python installation:  '**pip install refinitiv-data**'


  
## <a id="authors"></a>Authors
* **Marios Skevofylakas**
* **Haykaz Aramyan**
