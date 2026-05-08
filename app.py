import streamlit as st
import pickle
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="ReadRadar Recommender", layout="wide", initial_sidebar_state="collapsed")

# --- 2. PREMIUM CSS INJECTION ---
st.markdown("""
    <style>
    /* Force Dark Theme and Custom Fonts */
    .stApp {
        background-color: #1e1e1e;
        color: #eaeaea;
        font-family: 'Inter', sans-serif;
    }
    
    /* To Hide Streamlit Chrome & Sidebar completely */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}
    
    /* Top Navigation Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #1e1e1e;
        padding-top: 10px;
        padding-bottom: 10px;
        border-bottom: 1px solid #333;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #a0a0a0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #ff6bff !important;
        border-bottom: 2px solid #ff6bff !important;
    }
    
    /* Book Card Styling */
    div[data-testid="column"] {
        background-color: #252525;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #333;
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 20px;
    }
    div[data-testid="column"]:hover {
        transform: translateY(-5px);
        border-color: #4facfe;
    }
    
    /* Image container */
    img {
        border-radius: 8px;
        width: 100%;
        height: 250px;
        object-fit: cover;
    }
    
    /* Typography inside cards */
    .book-title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 12px;
        margin-bottom: 4px;
        line-height: 1.2;
        height: 2.4rem;
        overflow: hidden;
    }
    .book-author {
        font-size: 0.85rem;
        color: #aaaaaa;
        margin-bottom: 8px;
    }
    .book-stats {
        font-size: 0.8rem;
        color: #ffb703;
        font-weight: 600;
    }
    .book-votes {
        font-size: 0.75rem;
        color: #888888;
    }
    
    /* About Page Outline Boxes */
    .about-box {
        border: 1px solid #444;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #222;
    }
    .about-box h4 {
        color: #ffffff;
        margin-bottom: 15px;
        border-bottom: 1px solid #444;
        padding-bottom: 10px;
    }
    
    /* Global Search Styling */
    .stSelectbox label {
        font-size: 1.2rem;
        font-weight: bold;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING & HELPERS ---
@st.cache_resource
def load_models():
    pt = pickle.load(open('pt.pkl', 'rb'))
    books = pickle.load(open('books.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return pt, books, model

pt, books, model = load_models()
book_list = pt.index.values

def generate_stars(rating):
    """Converts a numerical rating into a star string."""
    rating = float(rating)
    # Scale from 0-10 rating to 0-5 stars
    scaled = round(rating / 2) 
    return "★" * scaled + "☆" * (5 - scaled)

def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    distances, indices = model.kneighbors(pt.iloc[index,:].values.reshape(1,-1), n_neighbors=6)
    
    data = []
    for i in indices.flatten()[1:]: 
        item = []
        temp_df = books[books['title'] == pt.index[i]]
        # Catching the exact details we need
        item.extend(list(temp_df.drop_duplicates('title')['title'].values))
        item.extend(list(temp_df.drop_duplicates('title')['author'].values))
        item.extend(list(temp_df.drop_duplicates('title')['image_url'].values))
        item.extend(list(temp_df.drop_duplicates('title')['avg_rating'].values))
        item.extend(list(temp_df.drop_duplicates('title')['num_ratings'].values))
        data.append(item)
    return data

def render_book_card(title, author, image_url, avg_rating, num_ratings):
    """HTML template for the book card to ensure consistent styling."""
    stars = generate_stars(avg_rating)
    st.markdown(f"""
        <img src="{image_url}" onerror="this.src='https://via.placeholder.com/150x200?text=No+Cover'">
        <div class="book-title">{title[:45]}{'...' if len(title) > 45 else ''}</div>
        <div class="book-author">{author}</div>
        <div class="book-stats">{stars} ({avg_rating:.1f}/10)</div>
        <div class="book-votes">{int(num_ratings)} total votes</div>
    """, unsafe_allow_html=True)


# --- 4. TOP NAVIGATION (Website Architecture) ---
tab1, tab2, tab3 = st.tabs(["Home", "Recommend", "About"])

# --- TAB 1: HOME PAGE ---
with tab1:
    st.markdown("## Trending Books")
    st.markdown("The most popular books currently actively rated by our community.")
    st.write("")
    
    # Sort books by number of ratings to get the most popular
    top_books = books.sort_values(by='num_ratings', ascending=False).head(10)
    
    cols = st.columns(5)
    for idx, row in enumerate(top_books.iterrows()):
        row_data = row[1]
        with cols[idx % 5]:
            render_book_card(
                row_data['title'], 
                row_data['author'], 
                row_data['image_url'], 
                row_data['avg_rating'], 
                row_data['num_ratings']
            )

# --- TAB 2: RECOMMEND PAGE (Global Search) ---
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h2 style='text-align: center;'>Find Your Next Read</h2>", unsafe_allow_html=True)
        selected_book = st.selectbox(
            "Search our database of over 1 million data points:", 
            book_list,
            index=None,
            placeholder="Type a book title here..."
        )
        
        search_btn = st.button('Search Similar Books', use_container_width=True)

    if search_btn and selected_book:
        st.markdown("---")
        st.markdown(f"### Because you liked **{selected_book}**...")
        
        with st.spinner('Calculating cosine similarities...'):
            recommendations = recommend(selected_book)
            
            rec_cols = st.columns(5)
            for i in range(5):
                with rec_cols[i]:
                    render_book_card(
                        recommendations[i][0], # Title
                        recommendations[i][1], # Author
                        recommendations[i][2], # Image
                        recommendations[i][3], # Avg Rating
                        recommendations[i][4]  # Num Ratings
                    )

# --- TAB 3: ABOUT PAGE (Inspired by uploaded image) ---
with tab3:
    st.markdown("## About This Project")
    st.markdown("A data-driven book recommender system that understands what readers love — built with precision, mathematics, and machine learning. Created By Shashank Upadhyay")
    
    colA, colB = st.columns([2, 1])
    
    with colA:
        st.markdown("""
        <div class="about-box">
            <h4>What is this Recommender?</h4>
            <p>This project blends <b>machine learning</b> with a modern UI to help readers discover new books they are likely to enjoy. It learns patterns from real rating behavior and uses mathematical similarity to deliver smart suggestions instantly.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="about-box">
            <h4>How the ML Model Works</h4>
            <ol style='color: #eaeaea; padding-left: 20px;'>
                <li style='margin-bottom: 10px;'><b>Data Prep:</b> Filter books with enough ratings to find meaningful patterns.</li>
                <li style='margin-bottom: 10px;'><b>User-Book Matrix:</b> Convert ratings into a grid that represents books as high-dimensional vectors.</li>
                <li style='margin-bottom: 10px;'><b>Cosine Similarity:</b> Measure the geometric angle between two books based on user behavior.</li>
                <li style='margin-bottom: 10px;'><b>Top Matches:</b> Rank the vectors and return the nearest neighbors.</li>
                <li style='margin-bottom: 10px;'><b>UI Serving:</b> Streamlit frontend renders the data dynamically.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <div class="about-box">
            <h4>Quick Info</h4>
            <div style='background: #333; padding: 8px; border-radius: 5px; margin-bottom: 8px; text-align: center;'>Collaborative Filtering</div>
            <div style='background: #333; padding: 8px; border-radius: 5px; margin-bottom: 8px; text-align: center;'>Cosine Similarity Matrix</div>
            <div style='background: #333; padding: 8px; border-radius: 5px; margin-bottom: 8px; text-align: center;'>Streamlit Frontend</div>
            <div style='background: #333; padding: 8px; border-radius: 5px; text-align: center;'>Pandas / NumPy</div>
        </div>
        """, unsafe_allow_html=True)