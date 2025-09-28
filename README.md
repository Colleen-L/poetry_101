# Poetry_101
## Description
The following is a search engine utilizing TF-IDF for searching poems in a poetry collection, _Poems Teachers Ask For_ produced by Charles Aldarondo and the Online Distributed Proofreading Team, that was found on Project Gutenberg. The search engine utilizes TF-IDF on the processed tokens, cosine similarity, and k-means clustering.

_Note_: Stemmed tokens are used for the cosine similarity search engine; lemmatized tokens are used for k-means clustering

The following steps are conducted:
1. Preprocessing the text
   - Separate the text into sections: intro, table of contents, main content (with poems), and ending
   - Extract title and author of each poem from the table of contents
   - Extract words from each poem in the main content
   - Connect the title and author to each poem

2. Compute TF-IDF
   - Calculate the term frequency
   - Calculate the inverse document frequency
   - Multiply the term frequency and the inverse document frequency for TF-IDF
   - Store the TF-IDF scores in a data frame

3. Search Functionality
   - Implement cosine similarity between the TF-IDF of the query and the dataframe
   - Stores results as a list to sort and return closes matches

4. K-Means Clustering
   - Utilized the lemmatized tokens
   - A random seed is chosen, centroids are intialized, centroids are updated (using cosine distances and calculating the mean vector of all poems in a cluster)
   - Process is repeated until convergence

The search can be conducted in Jupyter Notebook or in the React + FastAPI webapp for an improved user experience.
## Required Installations
These are required for all version of the project:
* ```jupyter notebook```
* ```python```
* ```pandas```
* ```numpy```
* ```nltk```
* ```re``` (usually preinstalled in Python)

They can be installed using the following command:
```pip install jupyter pandas numpy nltk```

If using the React + FastAPI version, addition installations are required:
Backend (FastAPI):
* ```fastapi```
* ```uvicorn```

```pip install fastapi uvicorn```

Frontend (React)

```npm install```
## Usage
### Jupyter Notebook
Prerequisites: Installation of Jupyter Notebook and the libraries listed in [Required Installations](#required-installations)

1. Run the cells in the notebook to preprocess and create the dataframes
2. Under the "SEARCH ENGINE FUNCTIONALITY" header, the following cell allows for changing the search query into the engine
3. After inputing a query, running the next few cells will provide the search output with the poems that match the keyword
4. The last few cells in the notebook, display the implementation of k-means clustering and the corresponding themes<sup>*</sup> that were produced

<sup>*</sup> _Note_: The clustering was overall inaccurate; the poems associated with the themes produced are likely inaccurate.
### React + FastAPI
Prerequisites: Installation of FastAPI, React, and the libraries listed in [Required Installations](#required-installations)

1. Run the FastAPI backend (developed using the code from Jupyter Notebook)

```uvicorn main:app --reload```

2. Run the React frontend

```npm run dev```

3. Navigate to the link provided (http://localhost:5173/)
4. Search using the search bar in the middle
5. Selecting the themes on the header will provide the poems associated with the following themes (from k-means clustering)
   - If theme is pressed, select "Clear Theme" to remove the theme

_Note_: A drawback of the webapp is that the search functionality and theme selection operate independently. Selecting a theme and then searching for a keyword will not provide a poem with the theme and the keyword; the keyword search will override the themes that appear.

## Final Findings

Search engines utilizing TF-IDF are helpful for keywords but falls short when determining sentiments and deeper meanings. The search engines succeed in returning searches where the themes are directly discussed and beneficial for finding ulterior, less accurate searches by optimizing cosine similarity. However, when implementing clustering for thematic ideas, the search engine falters. There were no clear k clusters value to produce optimal results, whether that is due to the poem sample size or due to the lack of sentiment analysis from TF-IDF. This was clear when testing with the elbow method; the graph took on the form of negative linear line.

* TF-IDF searching is useful for keyword searching and its adjacent texts in search engines.
* TF-IDF falls short in searching by sentiments, thematic ideas, and clustering.
* Clustering with TF-IDF scores does not result in clear and optimal k clusters.

## Resources (in APA)
_Cosine similarity – text similarity metric_ Study Machine Learning. https://studymachinelearning.com/cosine-similarity-text-similarity-metric/

Kavlakoglu, E., \& Winland, V. (2025, June 18). _What is K-means clustering?_ IBM. https://www.ibm.com/think/topics/k-means-clustering

Poems Teachers Ask For by Various. (2006, July 26). _Project Gutenberg_ https://www.gutenberg.org/ebooks/18909
