import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# 1. SETUP
_stemmer = StemmerFactory().create_stemmer()

NEGATION_WORDS = {"tidak", "bukan", "jangan", "tanpa", "belum", "gak", "ga"}
_default_stopwords = set(StopWordRemoverFactory().get_stop_words())
BASE_STOPWORDS = _default_stopwords - NEGATION_WORDS
EXTRA_STOPWORDS = {
    "yg", "ya", "nya", "sih", "dong", "deh", "kok", "kan", "aja", "nih",
    "gitu", "gt", "tdk", "utk", "dgn", "krn", "jd",
    "amp", "rt", "via",
}
ALL_STOPWORDS = (BASE_STOPWORDS | EXTRA_STOPWORDS) - NEGATION_WORDS

# 2. SLANG DICTIONARY
SLANG_DICT = {
    "gw": "saya","gue": "saya","gua": "saya","lu": "kamu","lo": "kamu","elu": "kamu",
    "gk": "tidak","ga": "tidak","gak": "tidak","g": "tidak","tdk": "tidak","nggak": "tidak","ngga": "tidak","engga": "tidak",
    "udh": "sudah","udah": "sudah","dah": "sudah",
    "blm": "belum","blom": "belum","sm": "sama","sma": "sama","bgt": "banget","bngt": "banget","yg": "yang","dgn": "dengan",
    "krn": "karena","karna": "karena","utk": "untuk","tp": "tapi","tpi": "tapi","jd": "jadi","jgn": "jangan",
    "gmn": "gimana","gmna": "gimana","knp": "kenapa","emg": "memang","emang": "memang","bkn": "bukan","trs": "terus","skrg": "sekarang","org": "orang",
    "hrs": "harus","sdh": "sudah","dr": "dari","krna": "karena","bs": "bisa","km": "kamu",
    "aq": "aku","ak": "aku","cm": "cuma","cuman": "cuma","pd": "pada","tdi": "tadi",
}

# 3. CLEANING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text) # Remove URL
    text = re.sub(r"@\w+", " ", text) # Remove mentions
    text = re.sub(r"#", " ", text) # Remove # but keep hashtag word
    text = re.sub(r"\d+", " ", text) # Remove numbers
    text = re.sub(r"[^\w\s]", " ", text) # Remove punctuation
    text = re.sub(r"(.)\1{2,}", r"\1\1", text) # Reduce repeated characters
    text = re.sub(r"\s+", " ", text).strip() # Remove extra spaces
    return text

# 4. TOKENIZATION
def tokenize(text):
    return text.split()

# 5. SLANG NORMALIZATION
def normalize_slang(tokens):
    return [SLANG_DICT.get(token, token) for token in tokens]

# 6. STOPWORD REMOVAL
def remove_stopwords(tokens):
    return [
        token
        for token in tokens
        if token not in ALL_STOPWORDS
    ]

# 7. STEMMING
_stem_cache = {}
def stem_tokens(tokens):
    result = []
    for token in tokens:
        if token not in _stem_cache:
            _stem_cache[token] = _stemmer.stem(token)
        result.append(_stem_cache[token])
    return result

# 8. FULL PREPROCESSING
def preprocess(text):
    text = clean_text(text) # Case folding + cleaning
    tokens = tokenize(text) # Tokenizing
    tokens = normalize_slang(tokens) # Slang normalization
    tokens = remove_stopwords(tokens) # Stopword removal
    tokens = stem_tokens(tokens) # Stemming
    # Remove single-character tokens
    tokens = [
        token
        for token in tokens
        if len(token) > 1
    ]
    return tokens

# 9. LOAD DATASET
url = ("https://raw.githubusercontent.com/ridife/dataset-idsa/refs/heads/master/Indonesian%20Sentiment%20Twitter%20Dataset%20Labeled.csv")
df = pd.read_csv(url, sep="\t")
df = df.rename(columns={"Tweet": "text","sentimen": "label"}) # Rename columns
label_map = {-1: "negative",0: "neutral",1: "positive"} # Convert labels
df["label"] = df["label"].map(label_map)
# 10. TAKE 2000 DATA FROM EACH LABEL
positive = df[df["label"] == "positive"].head(2000)
negative = df[df["label"] == "negative"].head(2000)
neutral = df[df["label"] == "neutral"].head(2000)
df_6000 = pd.concat([positive,negative,neutral],ignore_index=True)
# 11. PREPROCESS THE 6000 DATA
df_6000["tokens"] = df_6000["text"].apply(preprocess)
df_6000["clean_text"] = df_6000["tokens"].apply(lambda tokens: " ".join(tokens))
# 12. REMOVE EMPTY DATA
df_6000 = df_6000[df_6000["tokens"].map(len) > 0].reset_index(drop=True)
# 13. SAVE TO CSV
df_6000["tokens"] = df_6000["tokens"].apply(lambda tokens: " ".join(tokens))
df_6000.to_csv("preprocessed_Indonesian_Sentiment_Twitter_Dataset_Labeled_2000.csv",index=False)
print("Preprocessing selesai!\n")
print("Jumlah data:")
print(df_6000["label"].value_counts())      
print("\nData berhasil disimpan sebagai:", "\npreprocessed_Indonesian_Sentiment_Twitter_Dataset_Labeled_2000.csv")