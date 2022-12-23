
import streamlit as st 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from PIL import Image

# Load EDA
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer



# Load Our Dataset
def load_data(data):
	df = pd.read_csv(data)
	return df 

# Popular Books Function

def popular_books(df):
    rating_count=df.groupby("Book-Title").count()["Book-Rating"].reset_index()
    rating_count.rename(columns={"Book-Rating":"NumberOfVotes"},inplace=True)
    
    rating_average=df.groupby("Book-Title")["Book-Rating"].mean().reset_index()
    rating_average.rename(columns={"Book-Rating":"AverageRatings"},inplace=True)
    
    popularBooks=rating_count.merge(rating_average,on="Book-Title")
    
    def weighted_rate(x):
        v=x["NumberOfVotes"]
        R=x["AverageRatings"]
        
        return ((v*R) + (m*C)) / (v+m)
    
    C=popularBooks["AverageRatings"].mean()
    m=popularBooks["NumberOfVotes"].quantile(0.90)
    
    popularBooks=popularBooks[popularBooks["NumberOfVotes"] >=250]
    popularBooks["Popularity"]=popularBooks.apply(weighted_rate,axis=1)
    popularBooks=popularBooks.sort_values(by="Popularity",ascending=False)
    return popularBooks[["Book-Title","AverageRatings"]].reset_index(drop=True).head(50)


# COntent Based Filtering 
def content_based(bookTitle):
    bookTitle=str(bookTitle)
    
    if bookTitle in df["Book-Title"].values:
        rating_count=pd.DataFrame(df["Book-Title"].value_counts())
        rare_books=rating_count[rating_count["Book-Title"]<=100].index
        common_books=df[~df["Book-Title"].isin(rare_books)]
        
        if bookTitle in rare_books:
            most_common=pd.Series(common_books["Book-Title"].unique()).sample(3).values
            st.success(f'Could Find any good recommendation but here are some top picks') 
            st.success('Pick 1 :'+ most_common[0])
            st.success('Pick 2 :'+ most_common[1])
            st.success('Pick 3 :'+ most_common[2])   
        else:
            common_books=common_books.drop_duplicates(subset=["Book-Title"])
            common_books.reset_index(inplace=True)
            common_books["index"]=[i for i in range(common_books.shape[0])]
            targets=["Book-Title","Book-Author","Publisher"]
            common_books["all_features"] = [" ".join(common_books[targets].iloc[i,].values)for i in range(common_books[targets].shape[0])]
            vectorizer=CountVectorizer()
            common_booksVector=vectorizer.fit_transform(common_books["all_features"])
            similarity=cosine_similarity(common_booksVector)
            index=common_books[common_books["Book-Title"]==bookTitle]["index"].values[0]
            similar_books=list(enumerate(similarity[index]))
            similar_booksSorted=sorted(similar_books,key=lambda x:x[1],reverse=True)[1:6]
            books=[]
            for i in range(len(similar_booksSorted)):
                
                books.append(common_books[common_books["index"]==similar_booksSorted[i][0]]["Book-Title"].item())
      
            st.success('Pick 1 :'+ books[0])
            st.success('Pick 2 :'+ books[1])
            st.success('Pick 3 :'+ books[2])    
          
    else:
            st.success('Could not find any recommendation')                       




st.title("Book Recommendation App")

image = Image.open('Books.jpg')
st.image(image, caption='Pick your choice')

menu = ["Book lists","Recommendation Search","About"]
choice = st.sidebar.selectbox("Menu",menu)
df = load_data("Book_recommendation.csv")

if choice == "Book lists":
    st.subheader("Book lists")
    popular = popular_books(df)
    st.dataframe(popular)

elif choice == "Recommendation Search":
    st.subheader("Recommended Books")
    Book_you_have_read = st.text_input('Name of the book you liked reading:')
    if st.button("Recommend"):
            output_books = content_based(Book_you_have_read)
            
           

            
else:
    st.subheader("About")
    st.text("Built by Karthikeyan Nataraj")
    st.text("Data: Kaggle Book Recommendation Dataset")
    
