import streamlit as st
import pickle
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Read Radar | AI Book Engine", layout="wide", initial_sidebar_state="collapsed")

# --- 2. ADVANCED CSS INJECTION (Animations & Glow) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700;900&display=swap');

    /* Base Theme with deep radial background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1f1b2e 0%, #0b090f 100%);
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Hide the boring stuff */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}
    
    /* --- ANIMATIONS --- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(168, 85, 247, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(168, 85, 247, 0); }
        100% { box-shadow: 0 0 0 0 rgba(168, 85, 247, 0); }
    }

    /* Animated Gradient Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(270deg, #a855f7, #ec4899, #3b82f6, #a855f7);
        background-size: 300% 300%;
        animation: gradientShift 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0px;
    }
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 40px;
    }

    /* Top Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        background: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        color: #64748b;
        font-weight: 500;
        font-size: 1.2rem;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #fff !important;
        border-bottom: 3px solid #ec4899 !important;
        text-shadow: 0 0 15px rgba(236, 72, 153, 0.6);
    }
    
    /* --- GLASSMORPHISM BOOK CARDS --- */
    div[data-testid="column"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeInUp 0.7s ease-out forwards;
        opacity: 0; /* Starts hidden for animation */
    }
    div[data-testid="column"]:hover {
        transform: translateY(-10px) scale(1.02);
        background: rgba(255, 255, 255, 0.05);
        border-color: #a855f7;
        box-shadow: 0 15px 30px rgba(0,0,0,0.5), 0 0 20px rgba(168, 85, 247, 0.4);
    }
    
    /* Image styling inside columns */
    img {
        border-radius: 10px;
        width: 100%;
        height: 280px;
        object-fit: cover;
        box-shadow: 0 8px 16px rgba(0,0,0,0.6);
        margin-bottom: 12px;
    }
    
    /* Typography inside cards */
    .book-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
        line-height: 1.3;
        height: 2.8rem;
        overflow: hidden;
    }
    .book-author {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 300;
        margin-bottom: 12px;
    }
    .book-stats {
        font-size: 0.85rem;
        color: #fbbf24;
        font-weight: 700;
        letter-spacing: 2px;
    }
    .book-votes {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Input & Search Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ec4899 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 700;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
        animation: pulse 2s infinite;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.03);
        box-shadow: 0 0 20px rgba(236, 72, 153, 0.6);
        animation: none; /* Stop pulsing on hover */
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
    rating = float(rating)
    scaled = round(rating / 2) 
    return "★" * scaled + "☆" * (5 - scaled)

def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    distances, indices = model.kneighbors(pt.iloc[index,:].values.reshape(1,-1), n_neighbors=6)
    
    data = []
    for i in indices.flatten()[1:]: 
        item = []
        temp_df = books[books['title'] == pt.index[i]]
        item.extend(list(temp_df.drop_duplicates('title')['title'].values))
        item.extend(list(temp_df.drop_duplicates('title')['author'].values))
        item.extend(list(temp_df.drop_duplicates('title')['image_url'].values))
        item.extend(list(temp_df.drop_duplicates('title')['avg_rating'].values))
        item.extend(list(temp_df.drop_duplicates('title')['num_ratings'].values))
        data.append(item)
    return data

def render_book_card(title, author, image_url, avg_rating, num_ratings):
    stars = generate_stars(avg_rating)
    st.markdown(f"""
        <img src="{image_url}" onerror="this.src='https://via.placeholder.com/150x200/1f1b2e/ffffff?text=No+Cover'">
        <div class="book-title">{title[:40]}{'...' if len(title) > 40 else ''}</div>
        <div class="book-author">{author}</div>
        <div class="book-stats">{stars}</div>
        <div class="book-votes">{int(num_ratings)} readers · {avg_rating:.1f}/10</div>
    """, unsafe_allow_html=True)


# --- 4. TOP NAVIGATION & HERO SECTION ---
st.markdown('<h1 class="main-header">ReadRadar AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover your next universe. Powered by Machine Learning.</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">CREATED BY - Shashank Upadhyay</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔥 Trending", "🔍 Recommend", "🧠 The Math"])

# --- TAB 1: HOME PAGE ---
with tab1:
    st.write("")
    top_books = books.sort_values(by='num_ratings', ascending=False).head(10)
    
    # First Row
    cols1 = st.columns(5)
    for idx, row in enumerate(top_books.head(5).iterrows()):
        row_data = row[1]
        with cols1[idx]:
            render_book_card(row_data['title'], row_data['author'], row_data['image_url'], row_data['avg_rating'], row_data['num_ratings'])
            
    st.write("") # Spacing
    
    # Second Row
    cols2 = st.columns(5)
    for idx, row in enumerate(top_books.tail(5).iterrows()):
        row_data = row[1]
        with cols2[idx]:
            render_book_card(row_data['title'], row_data['author'], row_data['image_url'], row_data['avg_rating'], row_data['num_ratings'])

# --- TAB 2: RECOMMEND PAGE ---
with tab2:
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        selected_book = st.selectbox(
            "Target Book:", 
            book_list,
            index=None,
            placeholder="Type a book you loved..."
        )
        st.write("")
        search_btn = st.button('PULSE THE DATABASE', use_container_width=True)

    if search_btn and selected_book:
        st.markdown("---")
        st.markdown(f"<h3 style='text-align:center; color:#e2e8f0; font-weight:300;'>Because you interacted with <b style='color:#ec4899;'>{selected_book}</b></h3><br>", unsafe_allow_html=True)
        
        with st.spinner('Calculating vector distances...'):
            recommendations = recommend(selected_book)
            
            rec_cols = st.columns(5)
            for i in range(5):
                with rec_cols[i]:
                    render_book_card(
                        recommendations[i][0], 
                        recommendations[i][1], 
                        recommendations[i][2], 
                        recommendations[i][3], 
                        recommendations[i][4]  
                    )

# --- TAB 3: ABOUT PAGE ---
with tab3:
    st.write("")
    colA, colB = st.columns([2, 1])
    
    with colA:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); padding: 30px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05);">
            <h3 style="color:#ec4899; margin-top:0;">System Architecture</h3>
            <p style="color:#94a3b8; font-size:1.1rem; line-height:1.6;">
                This engine doesn't look at genres or summaries. It relies purely on human behavior. By transforming 1.1 million user ratings into a high-dimensional mathematical matrix, we represent every book as a vector. 
            </p>
            <p style="color:#94a3b8; font-size:1.1rem; line-height:1.6;">
                When you search for a book, the engine calculates the <b>Cosine Similarity</b>—measuring the geometric angle between the target book's vector and every other book in the database. The closest vectors represent books read by the exact same group of people.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); padding: 30px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); text-align:center;">
            <h4 style="color:#a855f7; margin-top:0;">Tech Stack</h4>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p style="color:#e2e8f0; font-weight:700; margin-bottom:5px;">Python & Pandas</p>
            <p style="color:#94a3b8; font-size:0.9rem;">Data engineering & matrix creation</p>
            <br>
            <p style="color:#e2e8f0; font-weight:700; margin-bottom:5px;">Scikit-Learn</p>
            <p style="color:#94a3b8; font-size:0.9rem;">K-Nearest Neighbors Algorithm</p>
            <br>
            <p style="color:#e2e8f0; font-weight:700; margin-bottom:5px;">Streamlit Cloud</p>
            <p style="color:#94a3b8; font-size:0.9rem;">Serverless hosting & UI</p>
        </div>
        """, unsafe_allow_html=True)