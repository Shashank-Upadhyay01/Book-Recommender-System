import streamlit as st
import pickle
import numpy as np

# Setting the page configuration for a professional wide layout
st.set_page_config(page_title="Book Recommender System", layout="wide", page_icon="📚")

# Loading the exported models
@st.cache_resource # This caches the models so they don't reload on every interaction
def load_models():
    pt = pickle.load(open('pt.pkl', 'rb'))
    books = pickle.load(open('books.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return pt, books, model

pt, books, model = load_models()

# App Header
st.title('📚 AI-Powered Book Recommendation Engine')
st.markdown("Discover your next favorite read. Select a book from the dropdown, and the Machine Learning model will find 5 visually similar recommendations based on collaborative filtering.")
st.divider()

# Sidebar for Search
st.sidebar.header("Search Parameters")
book_list = pt.index.values
selected_book = st.sidebar.selectbox("Type or select a book you enjoyed:", book_list)

def recommend(book_name):
    # Find the index of the selected book in the pivot table
    index = np.where(pt.index == book_name)[0][0]
    
    # Calculate the distances and indices of the nearest neighbors
    distances, indices = model.kneighbors(pt.iloc[index,:].values.reshape(1,-1), n_neighbors=6)
    
    data = []
    for i in indices.flatten()[1:]: 
        item = []
        # Getting the row from the original books dataframe
        temp_df = books[books['title'] == pt.index[i]]
        
        item.extend(list(temp_df.drop_duplicates('title')['title'].values))
        item.extend(list(temp_df.drop_duplicates('title')['author'].values))
        item.extend(list(temp_df.drop_duplicates('title')['image_url'].values))
        
        data.append(item)
    return data

# Triggering Recommendation
if st.sidebar.button('Generate Recommendations'):
    with st.spinner('Analyzing reading patterns...'):
        recommendations = recommend(selected_book)
        
        st.subheader(f"Because you liked **{selected_book}**, we recommend:")
        st.write("") 
        
        # Creating 5 columns for a clean, horizontal gallery layout
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                # Display Image
                st.image(recommendations[i][2], use_container_width=True)
                # Display Title and Author
                st.markdown(f"**{recommendations[i][0][:40]}...**")
                st.caption(f"✍️ {recommendations[i][1]}")