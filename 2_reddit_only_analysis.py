import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import bertopic
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import os
import streamlit as st
import pandas as pd
import praw
from dotenv import load_dotenv
from datetime import datetime
import re
import spacy
import sys
import openai
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from hdbscan import HDBSCAN
print("Interpreter in use:", sys.executable)
print("BERTopic version in use:", bertopic.__version__)

# --- Load Sentiment Model ---

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def classify_sentiment_score(texts):
    sentiment_scores = []
    for text in texts:
        encoded_input = tokenizer(
            text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = model(**encoded_input)
        scores = softmax(output.logits[0].detach().cpu().numpy())

        # Weighted sentiment: uses all three classes
        sentiment_score = (-1 * scores[0]) + (0 * scores[1]) + (1 * scores[2])
        sentiment_scores.append(sentiment_score)
    return sentiment_scores

# --- GPT Labeling Function ---


def generate_gpt_labels(df, topics_column, text_column):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    topic_labels = {}

    for topic_id in sorted(df[topics_column].unique()):
        if topic_id == -1:
            topic_labels[topic_id] = "Miscellaneous / Uncategorized"
            continue

        topic_df = df[df[topics_column] == topic_id]
        embeddings = np.stack(topic_df[embedding_col].values)
        centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        similarities = cosine_similarity(embeddings, centroid).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        sample_texts = topic_df.iloc[top_indices][text_column].tolist()
        sample_text_combined = "\n\n".join(sample_texts)

        prompt = f"""You are being shown 5 Reddit posts that have been grouped together because they are semantically similar.
                These posts all mention the same pet brand (either Chewy or Petco). Your task is to summarize the main theme or opinion expressed in these posts.
                Write a clear, short summary in under 12 words. Avoid vague labels like “customer experience” or “pets.”
                Don’t quote specific posts. Just describe the topic or sentiment they share.\n\n{sample_text_combined}"""

        try:
            response = openai.chat.completions.create(

                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=60,
            )
            label = response.choices[0].message.content.strip()
        except Exception as e:
            label = f"[Error generating label: {e}]"

        topic_labels[topic_id] = label

    return topic_labels


# Load Reddit API credentials from .env
load_dotenv()
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Initialize Reddit instance (read-only)
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Streamlit app layout
st.title("Reddit loves Chewy. Let's see how much")
st.markdown(
    "We read the Reddit threads so you don't have to. See what pet parents are really saying about Chewy")
st.markdown(" ")

# Sidebar inputs
st.sidebar.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)
st.sidebar.markdown("→  START HERE")


analysis_mode = st.sidebar.selectbox(
    "What would you like to explore?",
    ["Compare Chewy to a competitor", "Zoom in on Chewy (Coming Soon)"],
    index=0
)

competitor = st.sidebar.selectbox(
    "Who would you like to compare Chewy to?",
    ["Petco", "PetSmart"]
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

use_prefetched = st.sidebar.checkbox(
    "Use pre-fetched Reddit data (loads instantly – data as of May 28, 2025)",
    value=True
)


competitor = competitor.title()  # Ensures casing always matches summary keys

# Derive the selected model based on analysis mode
if analysis_mode == "Zoom in on Chewy (Coming Soon)":
    selected_analysis = "BERTopic"
elif analysis_mode == "Compare Chewy to a Competitor":
    selected_analysis = "Sentiment Analysis"
else:
    selected_analysis = None  # fallback in case of future expansion

brands = ["Chewy", competitor]

# Fetch posts when button is clicked
if st.sidebar.button("Fetch Posts"):

    if not use_prefetched and analysis_mode == "Compare Chewy to a competitor":
        # Clear old summaries to avoid cross-contamination from previous runs
        if "monthly_summaries" in st.session_state:
            del st.session_state["monthly_summaries"]

        try:
            all_brand_posts = []
            all_brand_comments = []

            for brand in brands:
                progress = st.progress(
                    0.0, text=f"Fetching {brand} posts...")
                brand_posts = []
                brand_comments = []

                if brand == "Chewy":
                    search_phrases = [
                        "chewy company",
                        "chewy.com",
                        "chewy dog",
                        "chewy cat",
                        "chewy order",
                        "chewy customer service",
                        "chewy refund",
                        "chewy flowers",
                        "chewy portrait",
                        "chewy autoship",
                        "chewy vet",
                        "chewy pharmacy"
                    ]
                    seen_ids = set()
                    chewy_submissions = []

                    for phrase in search_phrases:
                        progress.progress(search_phrases.index(phrase) / len(search_phrases) * 0.33,
                                          text=f"Fetching {brand} posts...")
                        results = reddit.subreddit("all").search(
                            query=phrase,
                            sort="relevance",
                            time_filter="year",
                            limit=100
                        )

                        for post in results:
                            chewy_submissions.append((post, phrase))
                    submissions = chewy_submissions

                else:
                    seen_ids = set()
                    competitor_submissions = []

                    results = reddit.subreddit("all").search(
                        query=brand,
                        sort="relevance",
                        time_filter="year",
                        limit=300
                    )

                    for i, post in enumerate(results):
                        if post.id not in seen_ids:
                            competitor_submissions.append(post)
                            seen_ids.add(post.id)
                        if i % 100 == 0:
                            progress.progress(min(i / 300, 1.0),
                                              text=f"Fetching {brand} posts...")

                    print(f"[{brand}] Total posts seen in loop: {i + 1}")
                    print(
                        f"[{brand}] After deduplication (unique post IDs): {len(competitor_submissions)}")

                    submissions = competitor_submissions

                if brand == "Chewy":
                    total_posts = len(submissions)
                    for post, phrase in submissions:
                        brand_posts.append({
                            "brand": brand,
                            "search_phrase": phrase,
                            "id": post.id,
                            "post_id": "N/A",
                            "timestamp": datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d"),
                            "text": post.title + " " + post.selftext,
                            "type": "post",
                            "score": post.score,
                            "upvote_ratio": post.upvote_ratio,
                            "num_comments": post.num_comments,
                            "url": post.url
                        })

                    # Convert Chewy posts to DataFrame and remove duplicates based on post ID
                    chewy_posts_df = pd.DataFrame(brand_posts)
                    chewy_posts_df = chewy_posts_df.drop_duplicates(
                        subset="id", keep="first")

                    # Filter out any rows that do not contain the word "chewy" in the text field (case-insensitive)
                    chewy_posts_df = chewy_posts_df[chewy_posts_df["text"].str.contains(
                        r"\bchewy\b", case=False, na=False)]

                    # Filter out posts mentioning Ryan Cohen, GameStop, or RC (case-insensitive)
                    chewy_posts_df = chewy_posts_df[
                        ~chewy_posts_df["text"].str.contains(
                            r"ryan cohen|gamestop|gme|\brc\b", case=False, na=False)
                    ]

                    # Drop rows where '[amazon]' appears in the text (case-insensitive)
                    chewy_posts_df = chewy_posts_df[~chewy_posts_df["text"].str.contains(
                        r"\[amazon\]", case=False, na=False)]

                    # Drop rows that mention Stella (case-insensitive)
                    chewy_posts_df = chewy_posts_df[~chewy_posts_df["text"].str.contains(
                        r"\bstella\b", case=False, na=False)]

                    # Filter chewy.com entries to only keep those that actually contain the string "chewy.com"
                    chewy_posts_df = chewy_posts_df[
                        ~((chewy_posts_df["search_phrase"] == "chewy.com") &
                          ~chewy_posts_df["text"].str.contains(r"chewy\.com", case=False, na=False))
                    ]

                    # Proceed to convert to records
                    brand_posts = chewy_posts_df.to_dict("records")

                    progress.progress(0.66, text=f"Filtering {brand} posts...")

                    # Pull top-level comments (up to 3) for filtered Chewy posts
                    total_rows = len(chewy_posts_df)
                    for i, (_, row) in enumerate(chewy_posts_df.iterrows()):
                        post = reddit.submission(id=row["id"])
                        post.comments.replace_more(limit=0)
                        top_comments = post.comments[:3]
                        for comment in top_comments:
                            brand_comments.append({
                                "brand": row["brand"],
                                "search_phrase": row["search_phrase"],
                                "id": comment.id,
                                "post_id": row["id"],
                                "timestamp": datetime.fromtimestamp(comment.created_utc).strftime("%Y-%m-%d"),
                                "text": comment.body,
                                "type": "comment",
                                "score": comment.score,
                                "url": f"https://reddit.com{comment.permalink}"
                            })

                        progress.progress(0.66 + ((i + 1) / total_rows)
                                          * 0.34, text=f"Fetching {brand} posts...")

                else:
                    for post in submissions:
                        phrase = brand
                        brand_posts.append({
                            "brand": brand,
                            "search_phrase": phrase,
                            "id": post.id,
                            "post_id": "N/A",
                            "timestamp": datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d"),
                            "text": post.title + " " + post.selftext,
                            "type": "post",
                            "score": post.score,
                            "upvote_ratio": post.upvote_ratio,
                            "num_comments": post.num_comments,
                            "url": post.url
                        })

                    # Convert competitor posts to DataFrame and remove duplicates based on post ID
                    brand_posts_df = pd.DataFrame(brand_posts)
                    brand_posts_df = brand_posts_df.drop_duplicates(
                        subset="id", keep="first")

                    print(
                        f"[{brand}] Posts before brand name text filter: {len(brand_posts_df)}")
                    print(
                        f"[{brand}] Posts after brand name text filter: {len(brand_posts_df)}")

                    # Filter out any rows that do not contain the competitor name in the text field (case-insensitive)
                    brand_posts_df = brand_posts_df[brand_posts_df["text"].str.contains(
                        fr"\b{brand.lower()}\b", case=False, na=False)]

                    # Drop rows where '[amazon]' appears in the text (case-insensitive)
                    brand_posts_df = brand_posts_df[~brand_posts_df["text"].str.contains(
                        r"\[amazon\]", case=False, na=False)]

                    # After filtering competitor posts
                    brand_posts = brand_posts_df.to_dict("records")

                    progress.progress(0.66, text=f"Filtering {brand} posts...")

                    # Pull top-level comments (up to 3) for filtered posts
                    total_rows = len(brand_posts_df)
                    for i, (_, row) in enumerate(brand_posts_df.iterrows()):
                        post = reddit.submission(id=row["id"])
                        post.comments.replace_more(limit=0)
                        top_comments = post.comments[:3]
                        for comment in top_comments:
                            brand_comments.append({
                                "brand": row["brand"],
                                "search_phrase": row["search_phrase"],
                                "id": comment.id,
                                "post_id": row["id"],
                                "timestamp": datetime.fromtimestamp(comment.created_utc).strftime("%Y-%m-%d"),
                                "text": comment.body,
                                "type": "comment",
                                "score": comment.score,
                                "url": f"https://reddit.com{comment.permalink}"
                            })

                        progress.progress(
                            0.66 + ((i + 1) / total_rows) * 0.34,
                            text=f"Fetching {brand} posts..."
                        )

                # Save each to session
                st.session_state[f"{brand}_posts"] = pd.DataFrame(brand_posts)
                st.session_state[f"{brand}_comments"] = pd.DataFrame(
                    brand_comments)

            # Clean Chewy comments before merging
            if "Chewy_comments" in st.session_state:
                chewy_comments_df = st.session_state["Chewy_comments"]
                chewy_comments_df = chewy_comments_df[chewy_comments_df["text"].str.contains(
                    r"\bchewy\b", case=False, na=False)]
                st.session_state["Chewy_comments"] = chewy_comments_df

            # Clean competitor comments before merging
            if f"{competitor}_comments" in st.session_state:
                competitor_comments_df = st.session_state[f"{competitor}_comments"]
                competitor_comments_df = competitor_comments_df[competitor_comments_df["text"].str.contains(
                    fr"\b{competitor.lower()}\b", case=False, na=False)]
                st.session_state[f"{competitor}_comments"] = competitor_comments_df

            # Merge all posts and comments
            all_posts = pd.concat(
                [st.session_state["Chewy_posts"], st.session_state[f"{competitor}_posts"]])
            all_comments = pd.concat(
                [st.session_state["Chewy_comments"], st.session_state[f"{competitor}_comments"]])
            df = pd.concat([all_posts, all_comments], ignore_index=True)
            df["brand"] = df["brand"].str.title()

            # Save final combined dataframe to session state for downstream analysis
            st.session_state["df"] = df

            with st.spinner("Just a moment..."):
                def classify_brand_comment(text, brand):
                    prompt = f"""Does the following Reddit post share a personal opinion or experience about {brand} the pet company?
                    We're looking for any post that could help us understand how customers feel about {brand} — positive, negative, or neutral.
                    Even brief opinions like “{brand} is great” should be included.
                    We welcome any length or style of opinion, as long as it helps us understand the customer's sentiment or experience.
                    Answer only with "yes" or "no."

                    {text}
                    """

                    try:
                        response = openai.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                            max_tokens=5,
                        )
                        label = response.choices[0].message.content.strip(
                        ).lower()
                        return brand if "yes" in label else f"Not {brand}"
                    except Exception as e:
                        return "Unknown"

                # Split dataframe into Chewy and competitor
                chewy_df = df[df["brand"] == "Chewy"].copy()
                competitor_df = df[df["brand"] != "Chewy"].copy()

                # Run GPT relevance classification on both Chewy and competitor data
                combined_df = pd.concat(
                    [chewy_df, competitor_df], ignore_index=True)
                relevance_results = []

                relevance_progress = st.progress(
                    0, text="Classifying brand relevance...")

                for i, row in combined_df.iterrows():
                    brand = row["brand"]
                    text = row["text"]
                    result = classify_brand_comment(text, brand)
                    relevance_results.append(result)
                    percent = int(((i + 1) / len(combined_df)) * 100)
                    relevance_progress.progress(
                        (i + 1) / len(combined_df),
                        text=f"Checking if each post actually talks about the brand... ({percent}%)"
                    )

                combined_df["brand_relevance"] = relevance_results

                # Keep only relevant Chewy posts
                chewy_df = combined_df[(combined_df["brand"] == "Chewy") &
                                       (combined_df["brand_relevance"] == "Chewy")]

                # Keep all competitor posts (no relevance filtering for now)
                competitor_df = combined_df[combined_df["brand"] != "Chewy"]

                competitor_df = combined_df[(combined_df["brand"] == competitor) &
                                            (combined_df["brand_relevance"]
                                            == competitor)
                                            ]

                # Recombine both into final dataset
                df = pd.concat([chewy_df, competitor_df], ignore_index=True)
                st.session_state["df"] = df

                st.session_state["df"] = df

                st.success(
                    f"Loaded {len(df)} items (posts + comments) across Chewy and {competitor}.")
                st.dataframe(df)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

    elif use_prefetched and analysis_mode == "Compare Chewy to a competitor":
        filename = f"data/chewy_{competitor.lower()}_prefetched.csv"
        df = pd.read_csv(filename)
        df["brand"] = df["brand"].str.title()

        # 🧼 Clear old summaries to prevent GPT from using wrong brand
        if "monthly_summaries" in st.session_state:
            del st.session_state["monthly_summaries"]

        st.session_state["df"] = df
        st.success(f"Loaded {len(df)} prefetched rows from CSV!")
        st.dataframe(df)

    elif analysis_mode == "Zoom in on Chewy (Coming Soon)":
        st.warning(
            "Zoom in on Chewy is coming soon — please select a comparison to fetch posts.")


# --- BERT and Sentiment Section ---

# Load spaCy and prepare stopwords
nlp = spacy.load("en_core_web_sm")
extra_stopwords = {
    "get", "like", "would", "even", "just", "really",
    "one", "around", "old", "time"
}
stop_words = nlp.Defaults.stop_words.union(extra_stopwords)


if "df" in st.session_state and not st.session_state["df"].empty:
    if st.button("Run Analysis"):
        selected_analysis = (
            "BERTopic" if analysis_mode == "Zoom in on Chewy (Coming Soon)" else "Sentiment Analysis")

        if "df" not in st.session_state or st.session_state["df"].empty:
            st.warning("No data to analyze. Please fetch posts first.")

        else:
            st.markdown("## Analysis Results")
            with st.spinner("Running Sentiment Analysis (this takes ~10 minutes — feel free to grab a coffee ☕)..."):

                # Pre Step A: Convert to DataFrame
                df = st.session_state.get("df")

                # Pre Step B: Combine title and selftext
                df["combined"] = df["text"].fillna('')

                # 3 case scenarios
                if selected_analysis is None:
                    st.error("Invalid analysis mode selected.")
                    st.stop()

                elif selected_analysis == "BERTopic":
                    # Run BERTopic modeling

                    # Step 3: Preprocess text (combined = title + selftext)
                    def preprocess_text(text, nlp, stop_words):
                        doc = nlp(text.lower())
                        return " ".join([
                            token.lemma_ for token in doc
                            if token.lemma_ not in stop_words and not token.is_punct and not token.is_space
                        ])

                    if "text_for_analysis" not in df.columns:
                        st.info("Preprocessing text for BERT model...")
                        df["text_for_analysis"] = df["combined"].apply(
                            lambda text: preprocess_text(text, nlp, stop_words)
                        )

                    # Save updated DataFrame with preprocessed text back to session
                    st.session_state["df"] = df

                    # Step 4: Run BERT
                    umap_model = UMAP(n_neighbors=10, random_state=42)
                    hdbscan_model = HDBSCAN(
                        min_cluster_size=5, prediction_data=True)

                    model = BERTopic(
                        umap_model=umap_model,
                        hdbscan_model=hdbscan_model,
                        calculate_probabilities=False,
                        verbose=False
                    )
                    topics, probs = model.fit_transform(
                        df["text_for_analysis"])
                    df["topic"] = topics

                    # Add this to generate embeddings for GPT labeling
                    embeddings = model.embedding_model.embed(
                        df["text_for_analysis"])
                    df["embedding"] = [vec for vec in embeddings]
                    embedding_col = "embedding"

                    # Generate GPT labels
                    gpt_labels = generate_gpt_labels(
                        df, topics_column="topic", text_column="text_for_analysis"
                    )

                    # Calculate total score and comment count per topic
                    topic_metrics = df.groupby("topic").agg(
                        Total_Score=("score", "sum"),
                        Total_Comments=("num_comments", "sum")
                    ).reset_index()

                    # Step 5: Show topics summary
                    topic_info = model.get_topic_info()
                    topic_info["GPT Label"] = topic_info["Topic"].map(
                        gpt_labels)

                    # Assign dominant brand to each topic
                    topic_info["Brand"] = df.groupby(
                        "topic")["brand"].agg(lambda x: x.mode()[0]).values

                    # Optional: Move Brand to the front of the columns
                    cols = ["Brand"] + \
                        [col for col in topic_info.columns if col != "Brand"]
                    topic_info = topic_info[cols]

                    # Merge in total score and total comment counts
                    topic_info = topic_info.merge(
                        topic_metrics, left_on="Topic", right_on="topic", how="left").drop(columns=["topic"])

                    st.write("### Topic Summary")
                    st.dataframe(topic_info)

                    # Step 6 (Optional): Visualize
                    st.write("### Topic Visualization")
                    fig = model.visualize_topics()
                    st.plotly_chart(fig)

                elif selected_analysis == "Sentiment Analysis":
                    # Run sentiment classification
                    df["sentiment_score"] = classify_sentiment_score(
                        df["combined"])

                    # Convert timestamp to datetime
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], errors="coerce")
                    df["month"] = df["timestamp"].dt.to_period("M")

                    # Count sentiment by brand and month
                    avg_sentiment = df.groupby(["brand", "month"])[
                        "sentiment_score"].mean().reset_index()

                    # Convert month back to datetime for plotting
                    avg_sentiment["month"] = avg_sentiment["month"].dt.to_timestamp()

                    import plotly.express as px
                    import streamlit.components.v1 as components

                    # --- BAR GRAPH: Total relevant post+comment count per brand per month ---
                    count_df = df.copy()
                    count_df["month"] = pd.to_datetime(
                        count_df["timestamp"]).dt.to_period("M").dt.to_timestamp()
                    monthly_counts = count_df.groupby(
                        ["brand", "month"]).size().reset_index(name="count")
                    bar_fig = px.bar(
                        monthly_counts, x="month", y="count", color="brand", barmode="group",
                        title="Monthly Volume of Brand Mentions (Reddit Posts + Comments)",
                        color_discrete_map={"Chewy": "#4B9CD3", "Petco": "#EF5B5B", "PetSmart": "#EF5B5B"})

                   # --- PIE CHART: % of Posts That Are Clearly Positive (as full pie) ---
                    tone_df = df.copy()

                    # Step 1: Label clearly positive posts
                    tone_df["tone"] = tone_df["sentiment_score"].apply(
                        lambda x: "Clearly Positive" if x > 0 else "Other"
                    )

                    # Step 2: Count by brand and tone
                    tone_counts = tone_df.groupby(
                        ["brand", "tone"]).size().reset_index(name="count")

                    # Step 3: Create separate pie DataFrames
                    chewy_pie_df = tone_counts[tone_counts["brand"] == "Chewy"]
                    competitor_pie_df = tone_counts[tone_counts["brand"]
                                                    == competitor]

                    # Step 4: Set consistent color map
                    pie_color_map = {
                        "Clearly Positive": "#8BCF88",
                        "Other": "#E0E0E0"
                    }

                    # Force tone order: Clearly Positive first, Other second
                    chewy_pie_df["tone"] = pd.Categorical(
                        chewy_pie_df["tone"], categories=["Clearly Positive", "Other"], ordered=True)
                    chewy_pie_df = chewy_pie_df.sort_values(
                        by="tone", ascending=True)

                    competitor_pie_df["tone"] = pd.Categorical(
                        competitor_pie_df["tone"], categories=["Clearly Positive", "Other"], ordered=True)
                    competitor_pie_df = competitor_pie_df.sort_values(
                        by="tone", ascending=True)

                    # Step 5: Plot Chewy Pie
                    chewy_pie = px.pie(
                        chewy_pie_df,
                        names="tone",
                        values="count",
                        title="Chewy: Sentiment Breakdown",
                        color="tone",
                        color_discrete_map=pie_color_map,
                        hole=0.3,
                        category_orders={"tone": ["Clearly Positive", "Other"]}
                    )

                    chewy_pie.update_traces(
                        texttemplate=[
                            f"{p:.1%}" if n == "Clearly Positive" else ""
                            for p, n in zip(chewy_pie_df["count"] / chewy_pie_df["count"].sum(), chewy_pie_df["tone"])
                        ]
                    )

                    # Step 6: Plot Competitor Pie
                    competitor_pie = px.pie(
                        competitor_pie_df,
                        names="tone",
                        values="count",
                        title=f"{competitor}: Sentiment Breakdown",
                        color="tone",
                        color_discrete_map=pie_color_map,
                        hole=0.3,
                        # <-- THIS is what controls the slice order
                        category_orders={"tone": ["Clearly Positive", "Other"]}
                    )

                    competitor_pie.update_traces(
                        texttemplate=[
                            f"{p:.1%}" if n == "Clearly Positive" else ""
                            for p, n in zip(competitor_pie_df["count"] / competitor_pie_df["count"].sum(), competitor_pie_df["tone"])
                        ]
                    )

                    # --- CONFIDENCE-WEIGHTED: Confidence-weighted score per month per brand ---

                    def extract_confidence_scores(texts):
                        all_scores = []
                        for text in texts:
                            encoded_input = tokenizer(
                                text, return_tensors='pt', truncation=True, max_length=512)
                            with torch.no_grad():
                                output = model(**encoded_input)
                            probs = softmax(
                                output.logits[0].detach().cpu().numpy())
                            all_scores.append({
                                "negative": probs[0],
                                "neutral": probs[1],
                                "positive": probs[2]
                            })
                        return pd.DataFrame(all_scores)

                    # Run only once and store in session
                    if "confidence_df" not in st.session_state:
                        st.session_state["confidence_df"] = extract_confidence_scores(
                            df["combined"])

                    df_conf = pd.concat(
                        [df.reset_index(drop=True), st.session_state["confidence_df"]], axis=1)
                    df_conf["month"] = pd.to_datetime(
                        df_conf["timestamp"]).dt.to_period("M").dt.to_timestamp()
                    df_conf["confidence_score"] = df_conf["positive"] - \
                        df_conf["negative"]

                    # --- GPT Monthly Summary: Create prompts for top/bottom sentiment posts ---

                    summary_inputs = []

                    for (brand, month), group in df_conf.groupby(["brand", "month"]):
                        print("==== Grouping ====")
                        print("Brand:", brand)
                        print("Month:", month)
                        print("Group size:", len(group))
                        print("Group sample sentiment:",
                              group["sentiment_score"].head(3).tolist())

                        group = group.copy()
                        group = group[~group["sentiment_score"].isna()]
                        # group = group[~group["confidence_score"].isna()]

                        # Filter by confidence threshold
                        # group = group[group["confidence_score"].abs() > 0.3]

                        # Top 3 and Bottom 3 with sentiment threshold
                        top_3 = group[group["sentiment_score"] > 0.2].sort_values(
                            by="sentiment_score", ascending=False).head(3)
                        bottom_3 = group[group["sentiment_score"] < -0.2].sort_values(
                            by="sentiment_score", ascending=True).head(3)

                        print("-- Filtered sentiment --")
                        print("Top 3 count:", len(top_3))
                        print("Bottom 3 count:", len(bottom_3))

                        texts = bottom_3["combined"].tolist(
                        ) + top_3["combined"].tolist()
                        label = ["Bottom 3"] * \
                            len(bottom_3) + ["Top 3"] * len(top_3)

                        if not texts:
                            continue  # skip empty cases

                        prompt = f"""You're analyzing Reddit posts about {brand} from {month.strftime('%B %Y')}.
                    The first {len(bottom_3)} posts are the most negative (Bottom 3), and the last {len(top_3)} are the most positive (Top 3).

                    Please summarize the overall sentiment using **safe-for-work, professional** bullet points, grouped under the following two sections.

                    **What People Liked**
                    - Identify 1–3 clear, concrete positive themes based on the Top 3 posts.
                    - Avoid vague summaries or generic praise. If the posts are neutral or unclear, say: "no strong praise this month".

                    **What People Disliked**
                    - Identify 1–3 complaints or concerns based on the Bottom 3 posts, but express them in a **measured, professional tone**.
                    - Avoid offensive language or extreme claims (e.g., "worst company ever").
                    - If the posts are neutral or unclear, say: "no strong complaints this month".

                    Reflect the *tone and intensity* of the posts without exaggerating. Keep it brief, clear, and respectful.

                    Here are the posts:
                    {chr(10).join(f"{l}: {t}" for l, t in zip(label, texts))}
                    """

                        summary_inputs.append({
                            "brand": brand,
                            "month": month,
                            "prompt": prompt
                        })

                    st.session_state["summary_prompts"] = summary_inputs

                    openai.api_key = os.getenv("OPENAI_API_KEY")
                    summaries = {}

                    for item in st.session_state["summary_prompts"]:
                        brand = item["brand"]
                        month = item["month"]
                        prompt = item["prompt"]

                        try:
                            response = openai.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.4,
                                max_tokens=300,
                            )
                            summary = response.choices[0].message.content.strip(
                            )
                        except Exception as e:
                            summary = f"[Error: {e}]"

                        summaries[(brand.title(), month.strftime(
                            '%Y-%m'))] = summary

                    st.session_state["monthly_summaries"] = summaries

                    if "monthly_summaries" not in st.session_state:
                        st.session_state["monthly_summaries"] = summaries
                    print("---- SUMMARY DICTIONARY KEYS ----")
                    print("Summary keys:", list(summaries.keys()))
                    print("Competitor variable is:", competitor)

                    trend_df = df_conf.groupby(["brand", "month"])[
                        "confidence_score"].mean().reset_index()
                    pivot_conf = trend_df.pivot(
                        index="month", columns="brand", values="confidence_score").sort_index()

                    # --- Display in horizontal scroll ---
                    st.markdown("### Compare Chewy to Competitor (4 Views)")

                    # View 1: Sentiment Breakdown (Pie Chart)
                    st.subheader("View 1: Sentiment Breakdown")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Chewy**")
                        st.plotly_chart(chewy_pie, use_container_width=True)
                    with col2:
                        st.markdown(f"**{competitor}**")
                        st.plotly_chart(
                            competitor_pie, use_container_width=True)

                    # View 2: Overall Sentiment Score (0–100 Scale)
                    st.subheader(
                        "View 2: Overall Sentiment Score (0–100 Scale)")

                    LOW_VOLUME_THRESHOLD = 10

                    # Identify valid months for the line chart only
                    monthly_total_counts = df.groupby("month").size()
                    valid_months_for_line = monthly_total_counts[monthly_total_counts >=
                                                                 LOW_VOLUME_THRESHOLD].index
                    valid_months_for_line = valid_months_for_line.to_timestamp()

                    # Normalize the sentiment score from [-1, 1] to [0, 100] for human-friendly readability
                    df["sentiment_percent"] = (
                        df["sentiment_score"] + 1) / 2 * 100

                    # Calculate monthly average percent by brand
                    df["month"] = pd.to_datetime(
                        df["timestamp"]).dt.to_period("M").dt.to_timestamp()
                    avg_sentiment_pct = df.groupby(["brand", "month"])[
                        "sentiment_percent"].mean().reset_index()
                    pivot_pct = avg_sentiment_pct.pivot(
                        index="month", columns="brand", values="sentiment_percent").sort_index()
                    pivot_pct = pivot_pct[pivot_pct.index.isin(
                        valid_months_for_line)]

                    # Plot
                    import matplotlib.pyplot as plt
                    fig_pct, ax_pct = plt.subplots(figsize=(8, 4))
                    pivot_pct["Chewy"].plot(
                        ax=ax_pct, label="Chewy", color="#4B9CD3", linewidth=2)
                    pivot_pct[competitor].plot(
                        ax=ax_pct, label=competitor, color="#EF5B5B", linewidth=2)

                    ax_pct.set_ylim(bottom=0, top=80)

                    for y in [10, 20, 30, 40, 60, 70]:
                        ax_pct.axhline(y, color="gray",
                                       linestyle="--", linewidth=0.5)
                        ax_pct.text(
                            x=pivot_pct.index[-1], y=y + 1, s=f"{y}", color="gray",
                            fontsize=8, ha='left', va='bottom'
                        )

                    ax_pct.set_title("Showing Only Months with 10+ Mentions")
                    ax_pct.set_ylabel("Sentiment Score (%)")
                    ax_pct.set_xlabel("Month")
                    ax_pct.axhline(50, color="black",
                                   linestyle="--", linewidth=0.5)

                    # Add sentiment direction labels
                    midpoint_month = pivot_pct.index[len(pivot_pct) // 2]
                    ax_pct.text(
                        x=midpoint_month, y=55,
                        s="Positive Sentiment ↑", fontsize=10,
                        color="black", ha='center', va='center'
                    )
                    ax_pct.text(
                        x=midpoint_month, y=45,
                        s="Negative Sentiment ↓", fontsize=10,
                        color="black", ha='center', va='center'
                    )

                    ax_pct.legend()

                    # Display
                    st.pyplot(fig_pct)

                    # View 3: Full Table of GPT Summaries
                    if "monthly_summaries" in st.session_state:
                        st.subheader(
                            "View 3: What People Liked and Disliked MOST Each Month")

                        st.markdown(
                            "*Note: These summaries highlight the most emotionally intense posts. "
                            "A month with overwhelmingly positive sentiment can still include a few strong complaints — and vice versa.*"
                        )

                        summaries = st.session_state["monthly_summaries"]

                        # st.markdown("**Debug: Summary Keys**")
                        # st.write(list(st.session_state.get(
                        #    "monthly_summaries", {}).keys()))

                        if summaries:
                            # Step 1: Group summaries by month
                            from collections import defaultdict

                            summaries_by_month = defaultdict(dict)
                            for (brand, month), text in summaries.items():
                                summaries_by_month[month][brand] = text

                            # Step 2: Interleave brands per month
                            table_data = []
                            for month in sorted(summaries_by_month.keys()):
                                brand_dict = summaries_by_month[month]

                                # Always show Chewy and selected competitor if available
                                ordered_brands = []
                                for b in ["Chewy", competitor]:
                                    if b in brand_dict:
                                        ordered_brands.append(b)

                                # Include any other brands if somehow present
                                ordered_brands += [
                                    b for b in brand_dict if b not in ordered_brands]

                                for brand in ordered_brands:
                                    table_data.append({
                                        "Brand": brand,
                                        "Month": pd.to_datetime(month).strftime('%B %Y'),
                                        "Summary": brand_dict[brand]
                                    })

                            # Display as expandable cards instead of raw table
                            for item in table_data:
                                label = f"{item['Brand']} — {item['Month']}"
                                with st.expander(label):
                                    st.markdown(item["Summary"])
                        else:
                            st.info(
                                "No summaries available. Run the analysis first.")

                    # View 4: Monthly Volume
                    st.subheader(
                        "View 4: Monthly Volume of Brand Mentions (Reddit Posts + Comments)")
                    st.plotly_chart(bar_fig, use_container_width=True)

                    st.session_state["analysis_complete"] = True
