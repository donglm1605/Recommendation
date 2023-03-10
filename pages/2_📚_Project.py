import streamlit as st



st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("https://i.pinimg.com/564x/9d/7d/67/9d7d6718e75261219a0de6f0f3cc7b6d.jpg");
            background-size: auto;
            }}
    </style>
    """, unsafe_allow_html=True)
choice = st.sidebar.selectbox('Chọn thông tin bạn cần xem',
                              ("Tổng quan dự án",
                               'CONTENT-BASED FILTERING',
                               'COLLABORATIVE FILTERING'))

# Tổng quan dự án
if choice == "Tổng quan dự án":
    st.markdown("<h2 style='text-align: left; color: #48D1CC;'>1. Tổng quan dự án</h2>",
                unsafe_allow_html=True)
    st.markdown("""* Tiki là một hệ sinh thái thương mại tất cả trong một, gồm các công ty thành viên như: Tikinow Smart Logistic ("TNSL"), Tiki Trading và trong số đó không thể không kể đến sàn thương mại điện tử Tiki.vn.
            """)
    st.markdown("""* Tiki.vn hiện đang là trang thương mại điện tử lọt top 2 tại Việt Nam và top 6 tại khu vực Đông Nam Á.
            """)
    st.markdown("""* Tiki.vn đã triển khai rất nhiều mô hình nhằm đáp ứng và cung cấp những trải nghiệm tốt nhất cho người tiêu dùng. 
                Và trong tương lai họ muốn xây dựng thêm nhiều công cụ hữu ích khác cũng như thu hút thêm nhiều user.
            """)
    st.markdown("""* Giả sử: Tiki.vn chưa xây dựng mô hình Recommendation System (Hệ thống đề xuất cho người tiêu dùng) tức là làm sao để người tiêu dùng có thể nhận được những gợi ý mua hàng phù hợp với nhu cầu của họ. 
            Cho nên, chúng tôi đã xây dựng mô hình Recommendation System trên sàn thương mại điện tử Tiki 
            bằng CONTENT-BASED FILTERING và COLLABORATIVE FILTERING để giải quyết vấn đề đó.
            """)
    st.image("image/tiki.png", width=700)
    st.markdown("<h2 style='text-align: left; color: #48D1CC;'>2. Recommender System</h2>",
                unsafe_allow_html=True)
    st.markdown("""* Recommender system là các thuật toán nhằm đề xuất các item có liên quan cho người dùng 
                (Item có thể là phim để xem, văn bản để đọc, sản phẩm cần mua hoặc bất kỳ thứ gì khác tùy thuộc 
                vào ngành dịch vụ).
                """)
    st.markdown("""* Recommender system thực sự quan trọng trong một số lĩnh vực vì chúng có thể tạo ra một khoản 
                thu nhập khổng lồ hoặc cũng là một cách để nổi bật đáng kể so với các đối thủ cạnh tranh.
                """)
    st.markdown("<h2 style='text-align: left; color: #48D1CC;'>3. Thuật toán sử dụng</h2>",
                unsafe_allow_html=True)
    st.image("image/thuattoan.png", width=700)

# CONTENT-BASED FILTERING
elif choice == "CONTENT-BASED FILTERING":
    st.markdown("<h1 style='text-align: center; color: #48D1CC;'>CONTENT-BASED FILTERING</h1>",
                unsafe_allow_html=True)
    st.markdown("""Gợi ý các item dựa vào hồ sơ (profiles) của người dùng hoặc dựa vào nội dung/thuộc tính (attributes) của những item tương tự như item mà người dùng đã chọn trong quá khứ. 
                Hai thuật toán phổ biến là Gensim và Cosine.
                """)
    st.image("image/contentbase.png", width=700)
    
    # Gensim
    st.markdown("## 1. Thuật toán Gensim - “Generate Similar”")
    st.markdown("""* Là một thư viện Python chuyên xác định sự tương tự về ngữ nghĩa giữa hai tài liệu thông qua mô hình không gian vector và bộ công cụ mô hình hóa chủ đề.
                """)
    st.markdown("""* Có thể xử lý kho dữ liệu văn bản lớn với sự trợ giúp của việc truyền dữ liệu hiệu quả và các thuật toán tăng cường
                """)
    
    # Link tham khảo
    url1 = 'https://www.tutorialspoint.com/gensim/index.htm'
    url2 = 'https://www.machinelearningplus.com/nlp/gensim-tutorial/'

    # Tạo nút dẫn tới link
    col1, col2, col3 = st.columns([1,1,2])

    with col1:
        st.markdown(""":blue[Link tham khảo:]""")
    with col2:
        st.markdown(f'''
        <a href={url1}><button style="background-color: #0000ff;color: white;">Click Here</button></a>
        ''',
        unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <a href={url2}><button style="background-color: #0000ff;color: white;">Click Here</button></a>
        ''',
        unsafe_allow_html=True)
        
    # Cosine
    st.markdown("## 2. Thuật toán Cosine - “Cosine-Similarity”")
    st.markdown("""* Ý tưởng chính của phương pháp này là đưa ra gợi ý dựa vào sự tương đồng với nhau giữa các sản phẩm.
                """)
    st.markdown("""* Một ưu điểm của Cosine Similarity là độ phức tạp thấp, 
                đặc biệt đối với các vectơ thưa thớt: chỉ cần xem xét các tọa độ khác không. Ví dụ, trong truy xuất thông tin và khai thác văn bản,
                mỗi từ được gán một tọa độ khác nhau và tài liệu được biểu diễn bằng vectơ số lần xuất hiện của mỗi từ. 
                Sau đó, Cosine Similarity đưa ra thước đo hữu ích về mức độ tương đồng của hai tài liệu, xét về chủ đề của chúng và không phụ thuộc vào độ dài của tài liệu.
                """)
    st.image("image/congthuc_cosine.png", width=700)
    
    # Link tham khảo
    url3 = 'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html'
    url4 = 'https://en.wikipedia.org/wiki/Cosine_similarity'

    # Tạo nút dẫn tới link
    col1, col2, col3 = st.columns([1,1,2])
    
    with col1:
        st.markdown(""":blue[Link tham khảo:]""")
    with col2:
        st.markdown(f'''
        <a href={url3}><button style="background-color: #0000ff;color: white;">Click Here</button></a>
        ''',
        unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <a href={url4}><button style="background-color: #0000ff;color: white;">Click Here</button></a>
        ''',
        unsafe_allow_html=True)
    
    
# COLLABORATIVE FILTERING
elif choice == "COLLABORATIVE FILTERING":
    st.markdown("<h1 style='text-align: center; color: #48D1CC;'>COLLABORATIVE FILTERING</h1>",
                unsafe_allow_html=True)
    st.image("image/collaborative.png", width=700)
    
    # Khái quát Collaborative Filtering
    st.markdown("## 1. Khái quát Collaborative Filtering")
    st.markdown("""* Gợi ý các items dựa trên sự tương quan (similarity) giữa các users và/hoặc items. 
                Có thể hiểu rằng đây là cách gợi ý tới một user dựa trên những users có hành vi tương tự (nó sử dụng kiến thức của số đông) 
                - (wisdom of the crowd).""")
    st.markdown("""* Collaborative Filtering (CF) được sử dụng nhiều hơn các Content - Based system 
                vì thường cho kết quả tốt hơn và tương đối dễ hiểu. Thuật toán có khả năng tự học feature, 
                nghĩa là nó có thể bắt đầu tự học những feature nào sẽ sử dụng.
                """)
    # Link tham khảo
    url5 = 'https://magz.techover.io/2021/12/25/neighborhood-based-collaborative-filtering-recommendation-systems-phan-3/'
    
    # Tạo nút dẫn tới link
    col1, col2, col3 = st.columns([1,1,2])
    
    with col1:
        st.markdown(""":blue[Link tham khảo:]""")
    with col2:
        st.markdown(f'''
        <a href={url5}><button style="background-color: #0000ff;color: white;">Click Here</button></a>
        ''',
        unsafe_allow_html=True)
    
    
    # Thuật toán ALS
    st.markdown("## 2. Thuật toán ALS")
    st.markdown("""* ALS về cơ bản là một cách tiếp cận Matrix Factorization (algorithm) để thực hiện recommendation algorithm 
                mà chúng ta phân tách ma trận user/item lớn -> các yếu tố user/item có chiều nhỏ hơn.
                """)
    st.markdown("""* Ví dụ:""")
    st.image("image/ALS.png", width=700)
    st.markdown("""* Tính toán các vị trí bị thiếu trong Rating Matrix:""")
    st.image("image/ALS2.png", width=700)