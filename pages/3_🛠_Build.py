import streamlit as st
import streamlit.components.v1 as com
from streamlit_lottie import st_lottie
import json

st.markdown(f"""
<style>
    .stApp {{
        background-image: url("https://i.pinimg.com/564x/9d/7d/67/9d7d6718e75261219a0de6f0f3cc7b6d.jpg");
        background-size: auto;
        }}
</style>
""", unsafe_allow_html=True)
menu = st.sidebar.selectbox('Chọn thuật toán bạn muốn xem',
                            ("Tiền xử lý" ,"Gensim", "Cosine", "ALS"))
if menu == "Tiền xử lý":
    st.markdown("<h1 style='text-align: center; color: #48D1CC;'>BUILD PROJECT</h1>",
                unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #f9db61;'>Tiền xử lý</h1>",
                unsafe_allow_html=True)
    # 1. Data show
    st.write("### 1. Show data")
    st.write("##### Data Product và biểu đồ biến Rating")
    st.image("image/productraw.png", width=900)
    st.image("image/ratingpr.png", width=900)
    st.write("##### Data Review và biểu đồ biến Rating")
    st.image("image/reviewraw.png", width=900)
    st.image("image/ratingre.png", width=900)
    
    # Kiểm tra outliers
    st.write("### 2. Kiểm tra outlier")
    st.write("##### Kiểm tra outlier của 2 data")
    st.image("image/outlier.png", width=900)
    st.image("image/outlier_pr_re.png", width=900)
    
    # Chọn lại các cột cần thiết
    st.write("### 3. Chọn lại các cột cần thiết")
    st.image("image/choice.png", width=900)
    
    # Kiểm tra null và Duplicates
    st.write("### 4. Kiểm tra null và duplicate")
    st.image("image/null_duplicate.png", width=900)
    
    # Data sau khi tiền xử lý
    st.write("### 4. Data sau khi xử lý text")
    st.image("image/preprocessing.png", width=900)
    
    
elif menu == "Gensim":
    st.markdown("<h1 style='text-align: center; color: #48D1CC;'>BUILD PROJECT</h1>",
                unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #f9db61;'>Gensim</h1>",
                unsafe_allow_html=True)
    # 1. Data show
    st.write("### 1. Show data prodcut")
    st.image("image/product.png", width=900)
    
    # 2. Vẽ WordCloud
    st.write("### 2. WordCloud")
    st.image("image/WC.png", width=900)
    
    # 3. Xây dựng model
    st.write("### 3. Xây dựng model")
    
    # 3.1
    st.write("##### 3.1 Viết lại hàm xử lý text")
    code1 = """def process_text(text):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    # Remove HTTP links
    document = regex.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', "", document)
    #Remove end of line characters
    document = regex.sub(r'[\ r \ n]+', " ", document)
    #Remove numbers, only keep letters
    document = regex.sub(r'[\w]*\d+[\w]*', " ", document)
    # Some lines start with a space, remove them
    document = regex.sub(r'^[\s]{1,}', " ", document)
    # Some lines end with a space, remove them
    document = regex.sub(r'[\s]{1,}$', " ", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        ###### DEL wrong words   
        new_sentence = new_sentence+ sentence + ' '                    
    document = new_sentence  
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

file = open('file/vietnamese_stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\ n')
file.close()"""
    st.code(code1, language='python')
    
    code2 = """
    product['final_content'] = product['content_clean'].apply(lambda x: word_tokenize(process_text(x), format='text'))
    product['content_clean'] = product.content_clean.apply(lambda x: str(x))
    product['final_content'] = product.final_content.apply(lambda x: str(x))
    products_gem = [[text for text in x.split()] for x in product.final_content]
    products_gem_re = [[t for t in text if not t in stopwords_lst] for text in products_gem]
    """
    st.code(code2, language='python')
    
    # 3.2
    st.write("##### 3.2 Tìm dictionary")
    code3 = "dictionary = corpora.Dictionary(products_gem_re)"
    st.code(code3, language='python')
    code4 = "feature_cnt = len(dictionary.token2id)"
    st.code(code4, language='python')
    
    # 3.3
    st.write("##### 3.3 Obtain corpus based on dictionary (dense matrix)")
    code5 = "corpus = [dictionary.doc2bow(text) for text in products_gem_re]"
    st.code(code5, language='python')
    
    # 3.4
    st.write("##### 3.4 Use TF-IDF Model to process corpus, obtaining index")
    code6 = "tfidf = models.TfidfModel(corpus)"
    st.code(code6, language='python')
    code7 = """index = similarities.SparseMatrixSimilarity(tfidf[corpus], 
                                            num_features = feature_cnt)"""
    st.code(code7, language='python')
    
    # 3.5
    st.write("##### 3.5 Viết hàm gợi ý sản phẩm")
    code8 = """def recommender(view_product, dictionary, tfidf):
    view_product = view_product.lower().split()
    kw_vector = dictionary.doc2bow(view_product)
    sim = index[tfidf[kw_vector]]
    
    list_id = []
    list_score = []
    
    for i in range(len(sim)):
        list_id.append(i)
        list_score.append(sim[i])
    df_result = pd.DataFrame({"id": list_id,
                              "score": list_score
                             })
    
    # five highest scores
    
    max_5_scores = df_result.sort_values(by="score", ascending=False)
    id_to_list = list(max_5_scores["id"])
    
    products_find = product[product.index.isin(id_to_list)]
    results = products_find[["item_id", "name", "image"]]
    results = pd.concat([results, max_5_scores], axis=1).sort_values(by="score", ascending=False)
    
    return results
    """
    
    st.code(code8, language='python')
    st.write("##### - Tính và in tất cả result")
    code9 = "results = recommender(name_description_pre, dictionary, tfidf)"
    st.code(code9, language='python')
    code10 = "results.head(20)"
    st.code(code10, language='python')
    st.write("##### - Tạo df final dựa trên df results")
    code29 = """final = results[["item_id", "id", "score"]]
    final.rename(columns = {'item_id':'product_id'}, inplace = True)
    final.rename(columns = {'id':'rcmd_product_id'}, inplace = True)
    """
    st.code(code29, language='python')
    code30 = "final.head(20)"
    st.code(code30, language='python')
    st.write("##### - Hàm cho người dùng nhập vào mô tả")
    code11 = """describe_text = input("- Nhập vào mô tả của bạn: ")
    num = int(input("- Bạn muốn xem bao nhiêu gợi ý: "))


    def result(doc, n):
        name_description_pre = describe_text
        number = num
        view_product = name_description_pre.lower().split()
        # Convert search words into Sparse Vectors
        kw_vector = dictionary.doc2bow(view_product)
        # Similarity calculation
        sim = index[tfidf[kw_vector]]
        # Sort sim 
        sim_sort = np.sort(sim)[::-1] # Descending (Giảm dần)
        top_index = sim.argsort()[-1*n:][::-1]
        # Print result
        for i in range(len(top_index)):
                print('- Keyword is similar to doc_index %d: %.2f' % (i, sim_sort[i]))
        for x in top_index:
                product_famillier = product["final_content"].iloc[[x]]
                print(product_famillier)
                print("*"*60)
        return i, product_famillier


    result = result(describe_text, num)"""
    st.code(code11, language='python')
    
    # 4. Lưu model
    st.write("### 4. Lưu model")
    codde28 = """final.to_csv("Data/contentbase.csv",
    index=False,
    header=True)"""
    st.code(codde28, language='python')
    
elif menu == "Cosine":
    st.markdown("<h1 style='text-align: center; color: #48D1CC;'>BUILD PROJECT</h1>",
                unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #f9db61;'>Cosine</h1>",
                unsafe_allow_html=True)
    # 1. Data show
    st.write("### 1. Show data prodcut")
    st.image("image/product.png", width=900)
    
    # 2. Đọc file vietnamese_stopwords
    st.write("### 2. Đọc file vietnamese_stopwords")
    code12 = """
    file =  open("file/vietnamese_stopwords.txt", 'r', encoding='utf-8')
    stop_words = file.read().split('\ n')"""
    st.code(code12, language='python')
    
    # 3. TFIDF và Tfidf_matrix
    st.write("### 3. TFIDF và Tfidf_matrix")
    
    # 3.1
    st.write("##### 3.1 TFIDF")
    code13 = "tf = TfidfVectorizer(analyzer='word', min_df=0, stop_words=stop_words)"
    st.code(code13, language='python')
    
    # 3.2
    st.write("##### 3.2 Tfidf_matrix")
    code14 = "tfidf_matrix = tf.fit_transform(product['content_clean'])"
    st.code(code14, language='python')
    
    # 4. Cosine_similarities
    
    # 4.1
    st.write("### 4. Cosine_similarities")
    st.write("##### 4.1 Cosine_similarities")
    code15 = "cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)"
    st.code(code15, language='python')
    
    # 4.2
    # Chuyển sang dataframe
    st.write("##### 4.2 Chuyển kết quả sang Dataframe")
    code16 = "df_matrix = pd.DataFrame(cosine_similarities)"
    st.code(code16, language='python')
    
    # 4.3
    # Tìm số ma trận
    st.write("##### 4.3 Tìm số ma trận của cosine")
    code17 = "print('- Số ma trận:',df_matrix.shape)"
    st.code(code17, language='python')
    
    # 5 Gợi ý người dùng
    st.write("### 5. Gợi ý người dùng")
    
    # 5.1
    st.write("##### 5.1 Nhập vào số gợi ý muốn xem")
    code18 = "n = int(input('- Bạn muốn xem bao nhiêu gợi ý: '))"
    st.code(code18, language='python')
    code19 = """# Function lấy n sản phẩm tương quan cao nhất
    def find_highest_score(cosine_similarities, nums):
        results = {}

        for idx, row in product.iterrows():    
            similar_indices = cosine_similarities[idx].argsort()[(-nums - 1) :- 1]
            similar_items = [(cosine_similarities[idx][i]) for i in similar_indices]
            similar_items = [(cosine_similarities[idx][i], product.index[i]) for i in similar_indices]
            print(similar_items[0:])
            results[idx] = similar_items[0:]
        return results

    result_cosine = find_highest_score(cosine_similarities, n)
    """
    st.code(code19, language='python')
    
    # 5.2
    st.write("##### 5.2 Nhập vào index muốn xem")
    code20 = "nhap = int(input('- Nhập vào index muốn xem:'))"
    st.code(code20, language='python')
    code21 = """def recomment_product(result_cosine, nhap):
                    i = nhap
                    recomment_1 = result_cosine[i][0][1]
                    print("- Phần trăm dự đoán với index 0: %.2f" % (result_cosine[i][0][0]))
                    print(product["content_clean"].iloc[recomment_1])
                    print("*"*60)
                    recomment_2 = result_cosine[i][1][1]
                    print("- Phần trăm dự đoán với index 1: %.2f" % (result_cosine[i][1][0]))
                    print(product["content_clean"].iloc[recomment_2])
                    print("*"*60)
                    recomment_3 = result_cosine[i][2][1]
                    print("- Phần trăm dự đoán với index 2: %.2f" % (result_cosine[i][2][0]))
                    print(product["content_clean"].iloc[recomment_3])
                    print("*"*60)
                    recomment_4 = result_cosine[i][3][1]
                    print("- Phần trăm dự đoán với index 3: %.2f" % (result_cosine[i][3][0]))
                    print(product["content_clean"].iloc[recomment_4])
                    print("*"*60)
                    recomment_5 = result_cosine[i][4][1]
                    print("- Phần trăm dự đoán với index 4: %.2f" % (result_cosine[i][4][0]))
                    print(product["content_clean"].iloc[recomment_5])
                    print("*"*60)
                    recomment_6 = result_cosine[i][5][1]
                    print("- Phần trăm dự đoán với index 5: %.2f" % (result_cosine[i][5][0]))
                    print(product["content_clean"].iloc[recomment_6])
                    print("*"*60)
                    recomment_7 = result_cosine[i][6][1]
                    print("- Phần trăm dự đoán với index 6: %.2f" % (result_cosine[i][6][0]))
                    print(product["content_clean"].iloc[recomment_7])
                    print("*"*60)
                    recomment_8 = result_cosine[i][7][1]
                    print("- Phần trăm dự đoán với index 7: %.2f" % (result_cosine[i][7][0]))
                    print(product["content_clean"].iloc[recomment_8])
                    print("*"*60)
                    recomment_9 = result_cosine[i][8][1]
                    print("- Phần trăm dự đoán với index 8: %.2f" % (result_cosine[i][8][0]))
                    print(product["content_clean"].iloc[recomment_9])
                    print("*"*60)
                    recomment_10 = result_cosine[i][9][1]
                    print("- Phần trăm dự đoán với index 9: %.2f" % (result_cosine[i][9][0]))
                    print(product["content_clean"].iloc[recomment_10])
                result_product = recomment_product(result_cosine, nhap)
                """
    st.code(code21, language='python')
    
    # 6 Gợi ý người dùng
    st.write("### 6. Lưu data")
    codde22 = "joblib.dump(cosine_similarities, 'Data/Cosine_similarities.pkl')"
    st.code(codde22, language='python')
    codde23 = "joblib.dump(result_cosine, 'Data/Result_cosine.pkl')"
    st.code(codde23, language='python')
    codde24 = "joblib.dump(dictionary, 'Data/Data/Dictionary.pkl')"
    st.code(codde24, language='python')
    codde25 = "joblib.dump(tfidf, 'Data/Tfidf.pkl')"
    st.code(codde25, language='python')
    codde26 = "joblib.dump(index, 'Data/Index.pkl')"
    st.code(codde26, language='python')
    codde27 = "joblib.dump(results, 'Data/Results.pkl')"
    st.code(codde27, language='python')
    
elif menu == "ALS":
    st.markdown("<h1 style='text-align: center; color: #48D1CC;'>BUILD PROJECT</h1>",
                unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #f9db61;'>ALS</h1>",
                unsafe_allow_html=True)
    # 1. Data show
    st.write("### 1. Show data review")
    st.image("image/review.png", width=900)
    
    # 2. Khám phá dữ liệu
    st.write("### 2. Khám phá dữ liệu")
    
    # 2.1
    st.write("##### 2.1 Tạo data_sub từ data")
    code31 = "data_sub = data.select(['customer_id', 'product_id', 'rating'])"
    st.code(code31, language='python')
    st.image("image/review.png", width=900)
    
    # 2.2
    st.write("##### 2.2 Số dòng của dữ liệu:")
    code32 = "print('- Số dòng của dữ liệu:', data_sub.count())"
    st.code(code32, language='python')
    st.write("- Số dòng của dữ liệu: 360211")
    
    # 2.3
    st.write("##### 2.3 Chuyển đổi kiểu dữ liệu phù hợp")
    code33 = "data_sub = data_sub.withColumn('rating', data_sub['rating'].cast(DoubleType()))"
    st.code(code33, language='python')
    
    # 2.4
    st.write("##### 2.4 Kiểm tra Null")
    code34 = """data_sub.select([count(when(col(c).isNull(), c)).alias(c) for c in
    data_sub.columns]).toPandas().T"""
    st.code(code34, language='python')
    st.write("- Dữ liệu không có giá trị null.")
    
    # 2.5 Đếm các giá trị duy nhất trong 3 biến (customer_id, product_id, rating)
    st.write("##### 2.5 Đếm các giá trị duy nhất trong 3 biến (customer_id, product_id, rating)")
    code35 = """
    users = data_sub.select("customer_id").distinct().count()
    products = data_sub.select("product_id").distinct().count()
    numerator = data_sub.count()
    ratings = data_sub.select("rating").distinct().count()
    """
    st.code(code35, language='python')
    st.write("- Customer_id: 251491")
    st.write("- Product_id: 4218")
    st.write("- Số lượng người dùng đã đánh giá: 360211")
    st.write("- Rating: 5")
    
    # 2.6 Đếm số ma trận xếp hạng có thể chứa nếu không có ô trống
    st.write("##### 2.6 Đếm số ma trận xếp hạng có thể chứa nếu không có ô trống")
    code36 = "denominator = users * products"
    st.code(code36, language='python')
    st.write("- Số ma trận: 1060789038")
    
    # 2.7 Tính độ thưa
    st.write("##### 2.7 Tính độ thưa")
    code37 = "denominator = users * products"
    st.code(code37, language='python')
    st.write("- Độ thưa: 0.9996604310686702")
    
    
    # 3. Triển khai ALS
    st.write("### 3. Triển khai ALS")
    code38 = """from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.recommendation import ALS
    from pyspark.ml import Pipeline
    from pyspark.sql.functions import col
    from datetime import datetime
    """
    st.code(code38, language='python')
    code39 = "(training, test) = data_sub.randomSplit([0.8, 0.2])"
    st.code(code39, language='python')
    
    # 4. Xây dựng model
    st.write("### 4. Xây dựng model")
    code40 = """start = datetime.now()
    als_1 = ALS(maxIter=5, # Số lần lặp
            regParam=0.09,
            rank = 10,
            userCol="customer_id",
            itemCol="product_id",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True)
    model_1 = als_1.fit(training)
    print("- Thời gian chạy:", datetime.now()-start)"""
    st.code(code40, language='python')
    
    # 5. Đánh giá kết quả
    st.write("### 5. Đánh giá kết quả")
    
    # 5.1
    st.write("##### 5.1 Model_1")
    code41 = "predictions_1 = model_1.transform(test)"
    st.code(code41, language='python')
    
    # RMSE
    st.write("##### - RMSE")
    code42 = """evaluator = RegressionEvaluator(metricName="rmse",
                                labelCol="rating",
                                predictionCol="prediction")
                                """
    st.code(code42, language='python')
    code43 = "rmse_1 = evaluator.evaluate(predictions_1)"
    st.code(code43, language='python')
    code44 = """print("Root-mean-square error = " + str(rmse_1))"""
    st.code(code44, language='python')
    st.write("Root-mean-square error = 2.039026591972255")
    
    # 5.2
    st.write("##### 5.2 Model_2")
    code45 = """from pyspark.ml.tuning import CrossValidator, ParamGridBuilder"""
    st.code(code45, language='python')
    code46 = """start = datetime.now()
                als_2 = ALS(userCol="customer_id",
                            itemCol="product_id",
                            ratingCol="rating",
                            coldStartStrategy="drop",
                            nonnegative=True)

                # Tạo parameters
                params = ParamGridBuilder().addGrid(als_2.regParam, [.01, .05, .1]).addGrid(als_2.rank, [5, 10, 50]).build()

                # Tạo crossValidator estimator
                cv = CrossValidator(estimator = als_2, estimatorParamMaps= params, evaluator= evaluator, parallelism= 4)
                best_model = cv.fit(data_sub)
                model_2 = best_model.bestModel

                print("- Thời gian chạy:", datetime.now()-start)
                """
    st.code(code46, language='python')
    
    code47 = """predictions_2 = model_2.transform(test)"""
    st.code(code47, language='python')
    
    # RMSE
    st.write("##### - RMSE")
    code48 = """evaluator = RegressionEvaluator(metricName="rmse",
                                labelCol="rating",
                                predictionCol="prediction")
                                """
    st.code(code48, language='python')
    code49 = "rmse_2 = evaluator.evaluate(predictions_2)"
    st.code(code49, language='python')
    code50 = """print("Root-mean-square error = " + str(rmse_2))"""
    st.code(code50, language='python')
    st.write("Root-mean-square error = 0.2562595335208489")
    
    # 6. Đưa ra đề xuất
    st.write("### 6. Đưa ra đề xuất")
    
    # 6.1
    st.write("##### 6.1 Nhận 10 đề xuất có xếp hạng cao nhất")
    code51 = """user_recs = model_2.recommendForAllUsers(10)"""
    st.code(code51, language='python')
    code52 = """for user in user_recs.head(5):
    print(user)
    """
    st.code(code52, language='python')
    
    # 6.2
    st.write("##### 6.2 Tạo df_customer_id chứa các phần tử duy nhất")
    code53 = """df_customer_id = data_sub.select('customer_id').distinct()"""
    st.code(code53, language='python')
    
    # 6.3
    st.write("##### 6.3 Tạo new_user_recs từ user_recs và df_customer_id")
    code54 = """new_user_recs = user_recs.join(df_customer_id, on=['customer_id'], how='left')"""
    st.code(code54, language='python')
    
    # 6.4
    st.write("##### 6.4 Đưa ra đề xuất dựa trên customer_id")
    code55 = """selected_users = [5235, 5678, 11763074]"""
    st.code(code55, language='python')
    code56 = """for user in selected_users:
    result = user_recs.filter(user_recs["customer_id"] == user)
    result.show(truncate=False)
    print("*"*234)
    """
    st.code(code56, language='python')
    
    # 7. Lưu kết quả
    st.write("### 7. Lưu kết quả")
    
    # 7.1
    st.write("##### 7.1 Pandas")
    code57 = """recs=model_2.recommendForAllUsers(10).toPandas()
                nrecs=recs.recommendations.apply(pd.Series) \
                    .merge(recs, right_index = True, left_index = True) \
                    .drop(["recommendations"], axis = 1) \
                    .melt(id_vars = ['customer_id'], value_name = "recommendation") \
                    .drop("variable", axis = 1) \
                    .dropna()
                nrecs=nrecs.sort_values('customer_id')
                nrecs=pd.concat([nrecs['recommendation']\
                    .apply(pd.Series), nrecs['customer_id']], axis = 1)
                nrecs.columns = [
                    'product_id',
                    'rating',
                    'customer_id']
    """
    st.code(code57, language='python')
    code58 = """md = data_sub.select(['customer_id', 'product_id'])
        md = md.toPandas()

        nrecs['customer_id']=nrecs['customer_id']
        nrecs['product_id']=nrecs['product_id']
        nrecs=nrecs.sort_values('customer_id')
        nrecs.reset_index(drop=True, inplace=True)
        """
    st.code(code58, language='python')
    code59 = """new = nrecs[['customer_id','product_id','rating']]"""
    st.code(code59, language='python')
    code60 = """new['recommendations'] = list(zip(new.product_id, new.rating))"""
    st.code(code60, language='python')
    code61 = """res=new[['customer_id','recommendations']]"""
    st.code(code61, language='python')
    code62 = """res_new=res['recommendations'].groupby([res.customer_id]).apply(list).reset_index()"""
    st.code(code62, language='python')
    code63 = """indices = range(len(res_new['recommendations'][0]))"""
    st.code(code63, language='python')
    code64 = """df_ = res_new['recommendations'].transform({f'SP{i+1}': itemgetter(i) for i in indices})"""
    st.code(code64, language='python')
    code65 = """df_merged = pd.concat([res_new, df_], axis=1)"""
    st.code(code65, language='python')
    
    # 7.2
    st.write("##### 7.2 Parquet")
    code66 = """new_user_recs.write.parquet('Recommend_User.parquet', mode='overwrite')"""
    st.code(code66, language='python')
    code67 = """df_product_id.write.parquet('Recommend_Product.parquet', mode='overwrite')"""
    st.code(code67, language='python')
    