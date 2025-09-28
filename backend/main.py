from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import copy
import pandas as pd
import numpy as np


app = FastAPI()

origins = [
  "http://localhost:5173",
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# normalizing helper function that removes punctuation and lowercasing
def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def title_and_author():
  with open("../poems_teachers_ask_for.txt", "r") as collection:
    text = collection.read()
  
  # extracts the table of contents for the poem title and poet name
  toc_start_indicator = "INDEX"
  toc_end_indicator = "PREFACE"
  toc_start_index = text.find(toc_start_indicator)
  toc_end_index = text.find(toc_end_indicator)

  toc = text[toc_start_index: toc_end_index]

  lines = toc.splitlines()
  basic_info = []
  for line in lines:
      res_entries = []
      # splits entries by author name
      entry = line.split("_")
      # splits by multiple spaces is author name is unknown
      if "_" not in line:
          entry = line.split("    ")
      for description in entry:
          # removes additional spaces
          clean_description = description.strip()
          if clean_description != "":
              res_entries.append(clean_description)
      #stories list of title, author, page in a list of entry lists
      basic_info.append(res_entries)
      
  # removes "INDEX" specifier
  del basic_info[0]
  # removes any empty list [] in basic_info
  basic_info = [entry for entry in basic_info if (len(entry) > 0) & (entry != [])]
  for entry in basic_info:
      # removes the last element of the entries (page number)
      entry.pop()
  
  #print(len(basic_info), len(cleaned_poems)) # 240 237 -> must normalize
  # populate unknown titles/authors with default descriptors
  normalized_titles = []
  titles = []
  authors = []

  for entry in basic_info:
    # both title and author exists
    if len(entry) == 2:
        title, author = entry
    # only title exists
    elif len(entry) == 1:
      title = entry[0]
      author = "Unknown"
    # both title and author doesn't exist (should not occur)
    else:
      title = "Untitled"
      author = "Unknown"

    # fix titles (e.g. "American Flag, The" -> "The American Flag")
    # finds titles with ending article
    match = re.match(r'(.+),\s*(The|A|An)$', title)
    if match:
      # swaps the main title and article
      main, article = match.groups()
      title = f"{article} {main}"

    # appends to lists
    normalized_titles.append(normalize(title))
    titles.append(title)
    authors.append(author)

  return authors, normalized_titles, titles
  
authors, normalized_titles, titles = title_and_author()

# load and preprocess data once on startup
def load_poems():
  with open("../poems_teachers_ask_for.txt", "r") as collection:
    text = collection.read()

  # extracts the main content: the collection of poems
  main_start_indicator = "O Captain! My Captain!"
  main_end_indicator = "*** END OF THE PROJECT GUTENBERG EBOOK"
  main_start_index = text.find(main_start_indicator)
  main_end_index = text.find(main_end_indicator)

  # the context of the collection: all of the poems
  text_content = text[main_start_index:main_end_index]


  stop_words = set(stopwords.words('english'))
  stemmer = PorterStemmer()
  lemmatizer = WordNetLemmatizer()

  stemmed_poems = []
  lemma_poems = []

  # every poem is separated by 4 or more new lines in a row
  poems = re.split(r'\n{4,}', text_content)
  # a to-be-cleaned version of the list of individual opens
  cleaned_poems = copy.deepcopy(poems)

  for i in range(len(cleaned_poems)):
    # remove the multiple new lines in a row
    cleaned_poems[i] = re.sub(r'\n{2,}', ' ', cleaned_poems[i])
    # replaces non-alphanumeric with space
    cleaned_poems[i] = re.sub(r'\W+', ' ', cleaned_poems[i])
    # removes underscores
    cleaned_poems[i] = cleaned_poems[i].replace('_', '')
    # replaces multiple spaces with a single space
    cleaned_poems[i] = re.sub(r'\s+', ' ', cleaned_poems[i])
    # separate each word to create a list // tokenization
    cleaned_poems[i] = cleaned_poems[i].split()

    # change all words to lowercase
    for j in range(len(cleaned_poems[i])):
      cleaned_poems[i][j] = cleaned_poems[i][j].lower()

    # remove empty entries in cleaned_poems

    cleaned_poems[i] = [word for word in cleaned_poems[i] if (len(word) > 0) & (word != [])]

    # remove stopwords with nltk
    cleaned_poems[i] = [word for word in cleaned_poems[i] if word not in stop_words]

    # stemming using nltk and storing in stemmed_poems list
    stemmed = [stemmer.stem(word) for word in cleaned_poems[i]]
    stemmed_poems.append(stemmed)

    # lemmatizing using nltk and storing in lemma_poems list
    lemma = [lemmatizer.lemmatize(word) for word in cleaned_poems[i]]
    lemma_poems.append(lemma)

  # Matching titles to poems

  # poems not split by words; lowercased and special characters removed
  normalized_poems = [normalize(''.join(poem)) for poem in poems]

  # match poems to the title by searching for title in each poem
  matched_poems = []
  matched_indices = []

  for title in normalized_titles:
    matched = False
    for id, poem in enumerate(normalized_poems):
      # checks if poem is already matched (using id)
      if id in matched_indices:
        continue
      if title in poem:
        matched_poems.append((title, id))
        matched_indices.append(id)
        matched = True
        break
    if not matched:
        matched_poems.append((title, None)) 

  # combine normalized titles, author, and poems (as list of words) into pandas data frame
  data = []

  for i, title in enumerate(titles):
    author = authors[i]

    # grabs index of poem in poems lists corresponding to title
    matched_index = matched_poems[i][1]

    if matched_index is not None:
      original_poem = poems[matched_index]
      cleaned = cleaned_poems[matched_index]
      stemmed = stemmed_poems[matched_index]
      lemmatized = lemma_poems[matched_index]

    data.append({
        "Title": title,
        "Author": author,
        "Original_Poem": original_poem,
        "Cleaned_Tokens": cleaned,
        "Stemmed_Tokens": stemmed,
        "Lemmatized_Tokens": lemmatized
    })

  # create dataframe
  df = pd.DataFrame(data)
  return df

df = load_poems()

def get_tfidf():
  # setting up tfidf dataframe

  # rows with id, title of the poem, and token
  rows = []

  for id, row in df.iterrows():
    # obtains information from the df
    tokens = row["Stemmed_Tokens"]
    title = row["Title"]
    total = len(tokens)
    for token in tokens:
      rows.append({"id": id, 'title': title, 'total_tokens': total, 'token': token})

  # creates a data frame with all of the tokens
  tokens_df = pd.DataFrame(rows)

  # group by id and token
  grouped = tokens_df.groupby(['id', 'token'])

  # counts each token and converts to dataframe
  tfidf_df = grouped.size().to_frame(name='count')
  # merge data frames (left) for total_tokens and title
  tfidf_df = tfidf_df.merge(tokens_df, on=['id', 'token'], how='left')

  #
  # computing term frequency (tf)
  #
  tfidf_df["tf"] = tfidf_df['count'] / tfidf_df['total_tokens']

  #
  # computing inverse term frequency (idf)
  #
  total_poems = df.shape[0] 

  # computes document frequency
  # group by token and determines number of unique documents/ids
  doc_f = tfidf_df.groupby('token')['id'].nunique() 
  # computes inverse document frequency of each token
  idf = np.log(total_poems / doc_f)

  # maps idf to tfidf dataframe using token
  tfidf_df['idf'] = tfidf_df['token'].map(idf)

  #
  # computing tf-idf
  #
  tfidf_df['tfidf'] = tfidf_df['tf'] * tfidf_df['idf']

  return idf, tfidf_df

idf, tfidf_df = get_tfidf()

@app.get("/search")
def search_poems(query: str = Query(..., min_length=1)):
  # removes non-alphanumeric characters
  query = re.sub(r'\W+', ' ', query)
  # replaces underscores
  query = query.replace('_', '')
  # reduces whitespaces to one space
  query = re.sub(r'\s+', ' ', query)
  # lowercase and splits words
  query_tokens = query.lower().split()

  # calculates the tf-idf of the query
  # computes terms with their corresponding counts
  query_tf = []
  query_tf_tokens = []
  for token in query_tokens:
    if token not in query_tf_tokens:
      query_tf_tokens.append(token)
      query_tf.append(1)
    else:
      index = query_tf_tokens.index(token)
      query_tf[index] += 1
  # creates dataframe for query tfidf
  query_tfidf = pd.DataFrame(query_tf, query_tf_tokens, columns=["count"])

  # computes query term frequency
  query_tfidf["tf"] = query_tfidf['count'] / sum(query_tfidf['count'])
  # finds corresponding idf for the token (default to 0 if not found)
  query_tfidf['idf'] = [idf.get(token, 0) for token in query_tf_tokens]

  # computes tfidf
  query_tfidf['tfidf'] = query_tfidf['tf'] * query_tfidf['idf']

  # helper function to calculate cosine similarity
  def cos_similarity(a, b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

  # implementing cosine similarity between tf-idf in query and every poems
  similarity_scores = []
  for poem_id in df.index:
    found_in_poem = []
    for token in query_tf_tokens:
      # finds the row in tfidf_df where query token appears in given poem
      row = tfidf_df[(tfidf_df['id'] == poem_id) & (tfidf_df['token'] == token)]
      if row.empty:
        found_in_poem.append(0)
      else:
        # access the tfidf of the token in the poem 
        found_in_poem.append(row.iloc[0]['tfidf'])

    # accounts for division by zero
    if np.linalg.norm(found_in_poem) == 0 or np.linalg.norm(query_tfidf['tfidf'].values) == 0:
      similarity = 0.0
    else:
      similarity = cos_similarity(found_in_poem, query_tfidf['tfidf'].values)

    similarity_scores.append({
      "id": poem_id,
      "title": df.loc[poem_id, "Title"],
      "author": df.loc[poem_id, "Author"],
      "poem": df.loc[poem_id, "Original_Poem"],
      "similarity": similarity
    })

  # sort similarity scores by obtaining similarity of each entry in similarity_scores
  similarity_scores = sorted(similarity_scores, key= lambda x: x['similarity'], reverse=True)
  top_results = similarity_scores[:10]

  return JSONResponse(content=top_results)


# creates a list of all the unique tokens
all_tokens = list(tfidf_df['token'].unique())
# assigns an index to each unique token
token_to_index = {}
for id, token in enumerate(all_tokens):
  token_to_index[token] = id

# number of poems
num_poems = df.shape[0]
# number of unique tokens
num_tokens = len(all_tokens)

# initialize the tfidf matrix with zeros
matrix = np.zeros((num_poems, num_tokens))

# populate the matrix
# loop over each row in the tfidf_df
for index, row in tfidf_df.iterrows():
  # obtain poem id
  poem_id = row['id']
  # obtain token
  token = row['token']
  # obtain tfidf score for the token
  tfidf_score = row['tfidf']

  if token in token_to_index:
    #find index corresponding to token
    token_index = token_to_index[token]

    # assign tfidf score to the correct location in matrix
    # x-coord: poem_id
    # y-coord: token_index

    matrix[poem_id, token_index] = tfidf_score  

# helper function to calculate cosine distance
def cosine_distances(A,B):
  # calculated distances between all rows of A and all rows of B

  # normalize
  A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
  B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)

  # compute cosine similarity
  similarity = np.dot(A_norm, B_norm.T)

  # return cosine distance
  return 1 - similarity


# initialize centroids
np.random.seed(23)
# IMPROVEMENT TO MAKE: find k value with elbow method and silhouette score...
k = 8
# chooses k random values from the matrix (centroids)
initial_centroids = matrix[np.random.choice(len(matrix), size=k, replace=False)]

# assign poems to nearest centroid
# compute Euclidean distance to each centroid
# reshapes and subtract centroids from each point
distances = cosine_distances(matrix, initial_centroids)
# finds index of minimum value in each row- closest centroid
labels = np.argmin(distances, axis=1)



def run_kmeans(matrix, k=8, seed=23):
    np.random.seed(seed)
    initial_centroids = matrix[np.random.choice(len(matrix), size=k, replace=False)]

    max_iters = 100
    threshold = 1e-4

    centroids = initial_centroids
    for iter in range(max_iters):
        distances = cosine_distances(matrix, centroids)
        labels = np.argmin(distances, axis=1)

        new_centroids = []
        for i in range(k):
            points = matrix[labels == i]
            if len(points) == 0:
                new_centroids.append(matrix[np.random.choice(len(matrix))])
            else:
                new_centroids.append(points.mean(axis=0))
        new_centroids = np.array(new_centroids)

        shift = np.linalg.norm(new_centroids - centroids)
        if shift < threshold:
            break
        centroids = new_centroids

    return labels

# Run clustering once on startup
labels = run_kmeans(matrix)
df["cluster"] = labels

cluster_to_themes = {
    0: "Childhood and Innocence",
    1: "Nature and New Beginnings",
    2: "Justice, Honor, and Humanity",
    3: "Rural life and Nature",
    4: "Patriotism and Youth",
    5: "Classical/Old Language",
    6: "Music and Romance",
    7: "Conflict and Labor"
}
themes_to_cluster = {v: k for k, v in cluster_to_themes.items()}

@app.get("/theme/{theme_name}")
def get_themes(theme_name: str):
    # Lookup the corresponding cluster number
    cluster_num = themes_to_cluster.get(theme_name)
    if cluster_num is None:
        return JSONResponse(content={"error": "Theme not found."}, status_code=404)

    # Get all poems in that cluster
    cluster_poems = df[df["cluster"] == cluster_num]

    # Build response
    response = []
    for _, row in cluster_poems.iterrows():
        response.append({
            "title": row["Title"],
            "author": row["Author"],
            "poem": row["Original_Poem"]
        })

    return JSONResponse(content=response)