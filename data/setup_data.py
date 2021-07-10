from pathlib import Path
import os

path = os.path.join(Path(os.path.dirname(__file__)).parent, "data")

class Data:

    user_data = os.path.join(path, "ml-100k", "u.data")

    movies = os.path.join(path, "ml-latest-small", "movies.csv")
    ratings = os.path.join(path, "ml-latest-small", "ratings.csv")
    links = os.path.join(path, "ml-latest-small", "links.csv")
    tags = os.path.join(path, "ml-latest-small", "tags.csv")

    online_retail = os.path.join(path, "online-retail", "online_retail.csv")
