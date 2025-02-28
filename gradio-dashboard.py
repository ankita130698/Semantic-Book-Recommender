import pandas as pd
import numpy as np
from dotenv import load_dotenv
from humanfriendly.terminal import output

# Import necessary LangChain components
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

# Load environment variables (e.g., API keys)
load_dotenv()

# Load the dataset containing book details and emotion scores
books = pd.read_csv('books_with_emotions.csv')

# Generate larger thumbnails for book covers (fallback to default cover if missing)
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "Book Cover.jpg",
    books["large_thumbnail"],
)

# Load and split text data for book descriptions (used for semantic search)
raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Create a semantic search database using Chroma and OpenAI embeddings
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

# Function to retrieve book recommendations based on query, category, and tone
def retrieve_semantic_recommendation(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,    # Function to retrieve book recommendations based on query, category, and tone
    final_top_k: int = 16,  # Number of final recommendations to display
) -> pd.DataFrame:

    # Perform semantic search on the book database
    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list = []
    for rec in recs:
        try:
            first_token = rec.page_content.strip().strip("'\"")  # Remove spaces, single, and double quotes
            first_token = first_token.split()[0]  # Extract first token (ISBN)
            books_list.append(int(first_token))  # Convert to integer
        except ValueError:
            print(f"Skipping invalid ISBN: {rec.page_content}")  # Debugging message

    # books_list = [int(rec.page_content.strip("'").split()[0]) for rec in recs]
    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    # Filter by category if not "All"
    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category].head(final_top_k)
    else:
        books_recs = books_recs.head(final_top_k)

    # Sort recommendations based on emotional tone if selected
    if tone == "Happy":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by="sadness", ascending=False, inplace=True)
    return books_recs

# Function to format and return book recommendations for display
def recommend_books(
        query: str,
        category: str,
        tone: str
):
        recommendations = retrieve_semantic_recommendation(query, category, tone)
        results = []

        for _, row in recommendations.iterrows():
            # Truncate long descriptions
            description = row["description"]
            truncated_desc_split = description.split()
            truncated_description = " ".join(truncated_desc_split[:30]) + "..."

            # Format author names
            authors_split = row["authors"].split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = row["authors"]

            # Create a caption for display
            caption = f"{row['title']} by {authors_str}: {truncated_description}"
            results.append((row["large_thumbnail"], caption))
        return results

# Define available categories and emotional tones
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Create Gradio UI for the book recommendation system
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Text(label = "Please enter a description of a book:", placeholder="e.g., A story about forgiveness")

        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone", value = "All")
        submit_button = gr.Button("Find recommendations")


    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    # Link button click to recommendation function
    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)

# Run the Gradio app
if __name__ == "__main__":
    dashboard.launch()
