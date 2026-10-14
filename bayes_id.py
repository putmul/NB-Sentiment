import pandas as pd
import streamlit as st

# TRAINING
@st.cache_data
def load_train_data():
    df = pd.read_csv('preprocessed_Indonesian_Sentiment_Twitter_Dataset_Labeled_2000.csv', sep=',')
    data = df['clean_text'].tolist()
    label = df['label'].tolist()

    # seperating words based on their appearence on a label
    positive, negative, neutral = [], [], []
    for sentence, lbl in zip(data, label):
        words = str(sentence).lower().split()

        if lbl == 'positive':
            positive.extend(words)
        elif lbl == 'negative':
            negative.extend(words)
        elif lbl == 'neutral':
            neutral.extend(words)

    # count unique words in the dataset
    unique_words = set()
    for sentence in data:
        for word in sentence.lower().split():
            unique_words.add(word)

    # Count the occurrences of each unique word in positive, negative, and neutral lists
    word_counts = {}
    for word in unique_words:
        word_counts[word] = {
            'positive': positive.count(word),
            'negative': negative.count(word),
            'neutral': neutral.count(word)
        }

    return (
        df,
        positive,
        negative,
        neutral,
        unique_words,
        word_counts
    )
df, positive, negative, neutral, unique_words, word_counts = load_train_data()

def kalkulasi(inputan):
    inputan_words = inputan.lower().split()

    positive_count = []
    negative_count = []
    neutral_count = []
    calculation_data = []

    # count the inputed words
    for word in inputan_words:
        if word in word_counts:
            positive_count.append(word_counts[word]['positive'])
            negative_count.append(word_counts[word]['negative'])
            neutral_count.append(word_counts[word]['neutral'])
        else:
            positive_count.append(0)
            negative_count.append(0)
            neutral_count.append(0)

    # calculate
    positive_value = 1.0
    negative_value = 1.0
    neutral_value = 1.0

    for i in range(len(inputan_words)):
        #               laplace smoothing    /   dominator
        p_positive = (positive_count[i] + 1) / (len(positive) + len(unique_words))
        p_negative = (negative_count[i] + 1) / (len(negative) + len(unique_words))
        p_neutral= (neutral_count[i] + 1) / (len(neutral) + len(unique_words))
        calculation_data.append({
            'Kata': inputan_words[i],
            'P(kata | Positive)': p_positive,
            'P(kata | Negative)': p_negative,
            'P(kata | Neutral)': p_neutral
            })
        positive_value *= p_positive
        negative_value *= p_negative
        neutral_value *= p_neutral

    if positive_value > negative_value and positive_value > neutral_value:
        sentiment = "Positive"
        st.success(f'😊 Sentimen kalimat "{inputan}" adalah **Positive**')
    elif negative_value > positive_value and negative_value > neutral_value:
        sentiment = "Negative"
        st.error(f'😠 Sentimen kalimat "{inputan}" adalah **Negative**')
    else:
        sentiment = "Neutral"
        st.info(f'😐 Sentimen kalimat "{inputan}" adalah **Neutral**')

    st.header("Detail Kalkulasi")
    calculation_df = pd.DataFrame(calculation_data)
    st.dataframe(calculation_df,use_container_width=True,hide_index=True)
    st.header("Hasil Kalkulasi")
    st.write("**Positive:**")
    st.code(f"{positive_value:.2e}")
    st.write("**Negative:**")
    st.code(f"{negative_value:.2e}")
    st.write("**Neutral:**")
    st.code(f"{neutral_value:.2e}")
    result_df = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Probability': [positive_value, negative_value, neutral_value]
        })
    st.bar_chart(result_df.set_index('Sentiment'))

def main():
    st.title("Sentiment Analysis Bahasa Indonesia")
    st.subheader("Streamlit Projects, Naive Bayes dengan Laplace Smoothing")
    inputan = st.text_area("Masukkan kalimat:",placeholder="Masukan teks kalimat...")
    if st.button("Analisis"):
        if inputan.strip():
            kalkulasi(inputan)
        else:
            st.warning(
                "Silakan masukkan kalimat terlebih dahulu."
            )
if __name__ == '__main__':
    main()
    