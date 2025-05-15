import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_csv("Books.csv")
users = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv")

df_clean = ratings[ratings["Book-Rating"] != 0]
ratings_with_title = df_clean.merge(books, on = "ISBN")
num_rating = ratings_with_title.groupby("Book-Title").count()["Book-Rating"]
num_rating = num_rating.reset_index()
num_rating.columns = ["Book-Title", "Num-Ratings"]
avg_rating = ratings_with_title.groupby("Book-Title").mean(numeric_only = True)["Book-Rating"].reset_index()
avg_rating.columns = ["Book-Title", "Avg-Ratings"]
popular_df = num_rating.merge(avg_rating, on = "Book-Title")
popular_df[popular_df["Num-Ratings"] >= 50].sort_values("Avg-Ratings", ascending = False).head(10)
x = ratings_with_title.groupby("User-ID").count() > 50
regular_users = x[x].index  #gets only the indexes of them
filtered_ratings = ratings_with_title[ratings_with_title["User-ID"].isin(regular_users)]
y = filtered_ratings.groupby("Book-Title")["Book-Title"].count()>= 30
famous_books = y[y].index
final_ratings = filtered_ratings[filtered_ratings["Book-Title"].isin(famous_books)]
piv_tab = final_ratings.pivot_table(index = "Book-Title", columns = "User-ID", values = "Book-Rating")
piv_tab.fillna(0, inplace= True)
similarity_scores = cosine_similarity(piv_tab)

#recommedation function

def recommend(book_name) :
  #index of the book_name from the pivot tabel
  index = np.where(piv_tab.index == book_name)[0][0]

  #getting the top 10 movies which are having high cosine similarity with the book
  similar_books = sorted(list(enumerate(similarity_scores[index])), key = lambda x : x[1], reverse= True)[1:11]

  #fetching the details of the books
  data = []
  for i in similar_books :
    item = []
    #fetching details of the similar books
    temp_df = books[books["Book-Title"] == piv_tab.index[i[0]]]

    #Adding title, author and image url
    item.extend(list(temp_df.drop_duplicates("Book-Title")["Book-Title"].values))
    item.extend(list(temp_df.drop_duplicates("Book-Title")["Book-Author"].values))
    item.extend(list(temp_df.drop_duplicates("Book-Title")["Image-URL-L"].values))

    data.append(item)

  return data

# ------------------------ STYLING ------------------------
st.set_page_config(page_title="Book Recommendation System", layout="centered")

st.markdown("""
    <style>
        /* Center header text and set color */
        h1 {
            text-align: center;
            color: #4B8BBE;
        }

        /* Style for input text box */
        div.stTextInput > div > input {
            font-size: 18px !important;
            padding: 10px !important;
            border: 2px solid #4B8BBE !important;
            border-radius: 8px !important;
        }

        /* Add some spacing */
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------ HEADER ------------------------
st.markdown("<h1>📚 Book Recommendation System</h1>", unsafe_allow_html=True)

# ------------------------ TEXT INPUT CENTERED ------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    book_input = st.text_input("Enter a book name you like")

# ------------------------ BUTTON CENTERED ------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    recommend_button = st.button("Recommend")

# ------------------------ RECOMMENDATION DISPLAY ------------------------
if recommend_button:
    if book_input in piv_tab.index:
        results = recommend(book_input)
        st.subheader("You might also enjoy:")

        for title, author, image in results:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {title}")
                st.markdown(f"*by {author}*")
            with col2:
                st.image(image, width=100)
            st.markdown("---")
    else:
        st.error("Book not found. Please try another title.")


