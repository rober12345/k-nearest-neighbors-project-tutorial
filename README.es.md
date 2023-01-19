<!-- hide -->
# k-nearest-neighbors Project Tutorial
<!-- endhide -->

- Al leer la lección de teoría de k-nearest neighbors, también leemos una introducción a los sistemas de recomendación. En este proyecto guiado, aprenderemos cómo construir un sistema de recomendación de películas simple utilizando el algoritmo de k-nearest neighbors.

- Este proyecto contiene 2 conjuntos de datos con características diferentes para las mismas 5000 películas, por lo que debes fusionarlos.

## 🌱  Cómo iniciar este proyecto

Esta vez no se hará Fork, tómate un tiempo para leer estas instrucciones:

1. Crear un nuevo repositorio basado en el [proyecto de Machine Learing](https://github.com/4GeeksAcademy/machine-learning-python-template/generate) [haciendo clic aquí](https://github.com/4GeeksAcademy/machine-learning-python-template).
2. Abre el repositorio creado recientemente en Gitpod usando la [extensión del botón de Gitpod](https://www.gitpod.io/docs/browser-extension/).
3. Una vez que Gitpod VSCode haya terminado de abrirse, comienza tu proyecto siguiendo las instrucciones a continuación.

## 🚛 Cómo entregar este proyecto

Una vez que hayas terminado de resolver los ejercicios, asegúrate de confirmar tus cambios, hazle "push" al fork de tu repositorio y ve a 4Geeks.com para subir el enlace del repositorio.

## 📝 Instrucciones

**Sistema de recomendación de películas**

¿Podemos predecir qué películas tendrán una calificación alta, incluso si no son un éxito comercial?

Este conjunto de datos es un subconjunto de la enorme API de base de datos de películas de TMDB, que contiene solo 5000 películas del número total.

Enlaces de conjuntos de datos:

tmdb_5000_movies: https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv

tmdb_5000_credits zip file to download: https://github.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/blob/main/tmdb_5000_credits.zip


**Paso 1:**

Importa las bibliotecas necesarias e importa el conjunto de datos.

```py
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

movies = pd.read_csv('../data/tmdb_5000_movies.csv')
credits = pd.read_csv('../data/tmdb_5000_credits.csv')
```

**Paso 2:**

Explora el conjunto de datos mirando las primeras filas y el número de filas y columnas.

```py
movies.head()

movies.shape

credits.head()

credits.shape
```

**Paso 3:**

Combina ambos marcos de datos en la columna 'título'.

```py
movies = movies.merge(credits, on='title')
```

**Paso 4:**

Trabajaremos únicamente con las siguientes columnas:

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

**Paso 5:**

Como solo faltan 3 valores en la columna "resumen", suéltelos.

```py
movies.isnull().sum()

movies.dropna(inplace = True)
```

**Paso 6:**

Como puedes ver, hay algunas columnas con formato json. Con el siguiente código, puedes ver qué géneros se incluyen en la primera fila.

```py
movies.iloc[0].genres

>>>>[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
```

Comenzaremos a convertir estas columnas usando una función para obtener solo los géneros, sin formato json. Solo estamos interesados ​​en los valores de las claves de 'nombre'.

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

Repite el proceso para la columna "palabras clave".

```py
movies['keywords'] = movies['keywords'].apply(convert)
```

Para la columna 'cast' crearemos una función nueva pero similar. Esta vez limitaremos el número de elementos a tres.

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

Puedes ver cómo va nuestro conjunto de datos:

```py
movies.head(1)
```

Las únicas columnas que quedan por modificar son 'crew' (equipo) y overview (resumen). Para el 'crew', crearemos una nueva función que permita obtener solo los valores de las claves 'name' (nombre) para cuyo 'job'(trabajo) el valor sea 'Director'. En resumen, estamos tratando de obtener el nombre del director.

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

Finalmente, veamos la primera fila de la columna 'resumen':

```py
movies.overview[0]
```

Para la columna 'overview', la convertiremos en una lista usando el método 'split()'.

```py
movies['overview'] = movies['overview'].apply(lambda x : x.split())
```

**Paso 7:**

Para que el sistema de recomendación no se confunda, por ejemplo, entre 'Jennifer Aniston' y 'Jennifer Conelly', quitaremos los espacios entre palabras con función.

```py
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
```

Ahora apliquemos nuestra función a las columnas 'genres' (géneros), 'cast' (elenco), 'crew' (equipo) y 'keywords' (palabras clave).

```py
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
```

**Paso 8:**

Reduciremos nuestro conjunto de datos combinando todas nuestras columnas convertidas anteriores en una sola columna llamada 'tags' (que crearemos).

Esta columna ahora tendrá TODOS los elementos separados por comas, pero ignoraremos las comas usando lambda x :" ".join(x).

```py
movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
```
```py
new_df = movies[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))
```

Mira cómo se ve ahora mostrando el primer tag:

```py
new_df['tags'][0]

>>>>'In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron'
```

**Paso 9:**

Usaremos el algoritmo KNN para construir el sistema de recomendación. Antes de ingresar al modelo, procedamos con la vectorización de texto que ya aprendiste en la lección de PNL.

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

Si deseas conocer las 5000 palabras más utilizadas, puede usar cv.get_feature_names().

**Paso 10:**

Encontremos la similitud de coseno entre las películas. Continúa y ejecuta las siguientes líneas de código en tu proyecto para ver los resultados.

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

**Paso 11:**

Finalmente, crea una función de recomendación basada en cosine_similarity. Esta función debería recomendar las 5 películas más similares.

```py
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] ##fetching the movie index
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate( distances)),reverse =True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
```

**Paso 12:**

Comprueba tu sistema de recomendación introduciendo una película. Corre a ver las recomendaciones.

```py
recommend('choose a movie here')
```

**Paso 13:**

Como siempre, usa notebook para experimentar y asegúrate de obtener los resultados que deseas.

Usa tu archivo app.py para guardar tus pasos definidos, canalizaciones o funciones en el orden correcto.

En tu archivo README escribe un breve resumen.

Guía de soluciones: 

https://github.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/blob/main/solution_guide.ipynb


