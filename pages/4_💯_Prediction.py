import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from st_clickable_images import clickable_images
import joblib


            

# Load data
@st.cache_data
def load_data(data):
    df = pd.read_csv(data)
    return df
df = load_data("data/Product_streamlit.csv")
recomend_dict = load_data("data/contentbase.csv")
#results = joblib.load("./Results.pkl")
#results = pd.DataFrame(results)
st.markdown(f"""
<style>
    .stApp {{
        background-image: url("https://i.pinimg.com/564x/9d/7d/67/9d7d6718e75261219a0de6f0f3cc7b6d.jpg");
        background-size: auto;
        }}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #48D1CC;'>USER PREDICTIONS</h1>",
            unsafe_allow_html=True)

menu = ["Tìm theo sản phẩm", "Tìm theo ID khách hàng"]
choice = st.sidebar.selectbox('Chọn phương thức mong muốn', menu)
# Contentbased
if choice == "Tìm theo sản phẩm":
    st.markdown("<h1 style='text-align: center; color: #f7e64c;'>Tìm theo sản phẩm</h1>",
                unsafe_allow_html=True)
    st.markdown("### 1. Nhập vào mô tả:")
    st.markdown("##### **:blue[- Hoặc lựa chọn bên dưới mục số 2 ⬇]** ")
    list_product = st.text_input(label=(""))
    # Lọc các mục dựa trên truy vấn của người dùng
    filtered_items = [item for item in df['name'] if list_product.lower() in item.lower()]
    # Hiển thị các mục đã lọc trong danh sách
    if filtered_items:
        st.markdown("### 2. Lựa chọn 1 trong các mô tả sau:")
        selected_item = st.selectbox("",filtered_items)
        st.markdown("##### **:blue[- Chọn số lượng sản phẩm muốn xem:]**")
        num_recommendation = st.slider("",
                                       min_value=0,
                                       max_value=10,
                                       step=1)
        selected_product_id = df[df['name'] == selected_item]['item_id'].values
        st.write(f"**Bạn chọn:** {selected_item}")
    else:
        st.write('Không tìm thấy sản phẩm đúng mô tả.')
        

    st.markdown("""### Các sản phẩm được đề xuất:""")
    recomend_items = recomend_dict[recomend_dict['product_id'].isin(selected_product_id)][['product_id', 'rcmd_product_id', 'score']].head(num_recommendation)
    recomend_items.sort_values(by=['score'], ascending=False, inplace=True)
        
    recomend_result = pd.merge(recomend_items, df, left_on='rcmd_product_id', right_on='item_id', how='left')
    recomend_result.sort_values(by=['rating' ,'score'], ascending=False, inplace=True)
        
    n_cols = 3
    n_rows = 1 + recomend_result.shape[0] // int(n_cols)
                        
    rows = [st.container() for _ in range(n_rows)]
                        
    cols_per_row = [r.columns(n_cols) for r in rows]
                        
    cols = [column for row in cols_per_row for column in row]
                        
    count = 0
                        
    for i, df in recomend_result.iterrows():
        image_url = df['image']
        product_name = df['name']
        price = "{:,.0f}".format(df['list_price'])
                            
        cols[count].image(df['image'],
                          width=150,
                          caption=df['name'],
                          use_column_width=True,
                          output_format="PNG")
        cols[count].write(f"**Brand:** {df['brand']}")
        cols[count].write(f"**Rating:** {df['rating']}")
        cols[count].write(f"**Giá:** {price}")
                

        count = count + 1
elif choice == "Tìm theo ID khách hàng":
    st.markdown("<h1 style='text-align: center; color: #f7e64c;'>Tìm theo ID khách hàng</h1>",
                unsafe_allow_html=True)
    
    def show_items(recommend_result):
        n_cols = 3
        n_rows = 1 + recommend_result.shape[0] // int(n_cols)
        
        rows = [st.container() for _ in range(n_rows)]
        cols_per_row = [r.columns(n_cols) for r in rows]
        cols = [column for row in cols_per_row for column in row]
        count = 0
        
        for i, df in recommend_result.iterrows():
            image_url = df['image']
            product_name = df['name']
            price = "{:,.0f}".format(df['list_price'])
        
            cols[count].image(df['image'],
                              width=150,
                              caption = df['name'],
                              use_column_width=True,
                              output_format="PNG")
            cols[count].write(f"**Brand:** {df['brand']}")
            cols[count].write(f"**Rating:** {df['rating_x']}")
            cols[count].write(f"**Giá:** {price}")
        

            count = count + 1
    
    def collaborative_filtering(show_items, recommend_dict, selected_item):
        st.markdown("""### 3. Các sản phẩm được đề xuất:""")
        recommend_items = recommend_dict.loc[recommend_dict['customer_id'] == selected_item, 'recommendations']
        recommend_items_lst = recommend_items.iloc[0]
        rcm_item_tuples_list = [(selected_item, item.get('product_id'), item.get('rating')) for item in recommend_items_lst]
        recommend_items_df = pd.DataFrame(rcm_item_tuples_list, columns=['customer_id', 'product_id', 'rating'])
        recommend_result = pd.merge(recommend_items_df, df, left_on='product_id', right_on='item_id', how='left')
        recommend_result.sort_values(by=['rating_x'], ascending=False, inplace=True)
        show_items(recommend_result)
    
    # Đọc data Review.csv và Recommend_User.parque
    data = pd.read_csv('data/Review.csv')
    recommend_dict = pd.read_parquet('data/Recommend_User.parquet')
    
    st.markdown("### 1. Nhập vào ID khách hàng muốn xem:")
    st.markdown("##### **:blue[- Hoặc lựa chọn bên dưới mục số 2 ⬇]** ")
    search_input = st.text_input('')
    filtered_items = [item for item in recommend_dict['customer_id'] if str(search_input).lower() in str(item).lower()]
    if filtered_items:
        st.markdown("### 2. Chọn ID khách hàng muốn xem:")
        selected_item = st.selectbox('', filtered_items)
        st.write(f"Bạn chọn ID: {selected_item}")
        collaborative_filtering(show_items, recommend_dict, selected_item)
    else:
        selected_item = st.selectbox('', filtered_items)
        st.write('Không có khách hàng cần tìm')
    
