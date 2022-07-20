<!-- hide -->
# k-nearest-neighbors Project Tutorial
<!-- endhide -->

- When reading the k-nearest neighbors theory lesson, we also read an introduction to recommender systems. In this guided project we will learn how to build a simple movie recommender system using k-nearest neighbors algorithm.

- This project contains 2 datasets with different features for the same 5000 movies, so you should merge them.

## 🌱  How to start this project

You will not be forking this time, please take some time to read this instructions:

1. Create a new repository based on [machine learning project](https://github.com/4GeeksAcademy/machine-learning-python-template/generate) by [clicking here](https://github.com/4GeeksAcademy/machine-learning-python-template).
2. Open the recently created repostiroy on Gitpod by using the [Gitpod button extension](https://www.gitpod.io/docs/browser-extension/).
3. Once Gitpod VSCode has finished opening you start your project following the Instructions below.

## 🚛 How to deliver this project

Once you are finished creating your movie recommender system, make sure to commit your changes, push to your repository and go to 4Geeks.com to upload the repository link.


## 📝 Instructions

**Movie recommender system**

Can we predict which films will be highly rated, even if they are not a commercial success?

This dataset is a subset of the huge TMDB Movie Database API, containing only 5000 movies from the total number.

Dataset links:

tmdb_5000_movies: https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv

tmdb_5000_credits zip file to download: https://github.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/blob/main/tmdb_5000_credits.zip


**Step 1:**

Import the necessary libraries and import the dataset.

```py
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

movies = pd.read_csv('../data/tmdb_5000_movies.csv')
credits = pd.read_csv('../data/tmdb_5000_credits.csv')
```

**Step 2:**

Explore the dataset by looking at the first rows and the number of rows and columns.

```py
movies.head()

movies.shape

credits.head()

credits.shape
```

**Step 3:**

Merge both dataframes on the 'title' column.

```py
movies = movies.merge(credits, on='title')
```

**Step 4:**

We will work only with the following columns:

-movie_id
-title
-overview
-genres
-keywords
-cast
-crew

```py
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
```

**Step 5:**

As there are only 3 missing values in the 'overview' column, drop them.

```py
movies.isnull().sum()

movies.dropna(inplace = True)
```

**Step 6:**

As you can see there are some columns with json format. With the following code, you can view what genres are included in the first row.

```py
movies.iloc[0].genres

>>>>[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
```

We will start converting these columns using a function to obtain only the genres, without a json format. We are only interested in the values of the 'name' keys.

```py
import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
```
```py
movies.dropna(inplace = True)
```
```py
movies['genres'] = movies['genres'].apply(convert)
movies.head()
```

Repeat the process for the 'keywords' column.

```py
movies['keywords'] = movies['keywords'].apply(convert)
```

For the 'cast' column we will create a new but similar function. This time we will limit the number of items to three.

```py
def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
        count +=1  
    return L
```
```py
movies['cast'] = movies['cast'].apply(convert3)
```

You can see how our dataset is coming along:

```py
movies.head(1)
```

The only columns left to modify are 'crew' and 'overview'. For the 'crew', we will create a new function that allows to obtain only the values of the 'name' keys for whose 'job' value is 'Director'. To sum up, we are trying to get the name of the director.

```py
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
```

```py
movies['crew'] = movies['crew'].apply(fetch_director)
```

Finally, let's look at the first row of the 'overview' column:

```py
movies.overview[0]
```

For the 'overview' column, we will convert it in a list by using 'split()' methode.

```py
movies['overview'] = movies['overview'].apply(lambda x : x.split())
```

**Step 7:**

For the recommender system to do not get confused, for example between 'Jennifer Aniston' and 'Jennifer Conelly', we will remove spaces between words with a function.

```py
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
```

Now let's apply our function to the 'genres', 'cast', 'crew' and 'keywords' columns.

```py
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
```

**Step 8:**

We will reduce our dataset by combining all our previous converted columns into only one column named 'tags' (which we will create).
This column will now have ALL items separated by commas, but we will ignore commas by using lambda x :" ".join(x).

```py
movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
```
```py
new_df = movies[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))
```

Look how it looks now by showing the first tag:

```py
new_df['tags'][0]

>>>>'In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron'
```

**Step 9:**

We will use KNN algorithm to build the recommender system. Before entering the model let's proceed with the text vectorization which you already learned in the NLP lesson.

```py
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000 ,stop_words='english')
```
```py
vectors = cv.fit_transform(new_df['tags']).toarray()
```
```py
vectors.shape
```

If you wish to know the 5000 most frequently used words you can use cv.get_feature_names()

**Step 10:**

Let's find the cosine_similarity among the movies. Go ahead and run the following code lines in your project to see the results.

```py
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vectors).shape
```
```py
similarity = cosine_similarity(vectors)
```
```py
similarity[0]
```
```py
sorted(list(enumerate(similarity[0])),reverse =True , key = lambda x:x[1])[1:6]
```

**Step 11:**

Finally, create a recommendation function based on the cosine_similarity. This function should recommend the 5 most similar movies.

```py
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] ##fetching the movie index
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate( distances)),reverse =True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
```

**Step 12:**

Check your recommender system by introducing a movie. Run to see the recommendations.

```py
recommend('choose a movie here')
```

**Step 13:**

As always, use your notebook to experiment and make sure you are getting the results you want. 

Use you app.py file to save your defined steps, pipelines or functions in the right order. 

In your README file write a brief summary.

Solution guide: 

https://github.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/blob/main/solution_guide.ipynb


