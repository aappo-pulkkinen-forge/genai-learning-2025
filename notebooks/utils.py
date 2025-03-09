import pandas as pd
import os
import requests
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
)


def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded: {save_path}")
        else:
            print(f"Failed to download {url} - Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def load_image_data(tsv_file, output_folder, n_images: int = 10000):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Load TSV file into DataFrame
    df = pd.read_csv(tsv_file, sep="\t")

    # Sort by photo_width
    df_sorted = df.sort_values(by="photo_width", ascending=True)

    # Get the first 1000 image URLs
    top_images = df_sorted.head(n_images)

    # Download images using multiple threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        for index, row in top_images.iterrows():
            image_url = row["photo_image_url"]
            image_name = os.path.join(output_folder, f"image_{index}.jpg")
            executor.submit(download_image, image_url, image_name)

    print("Download complete!")


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def generate_embeddings(image_model, image_folder, pickle_file, embeddings):
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            embeddings = pickle.load(f)
        print("Loaded embeddings from pickle file.")
        return embeddings
    image_files = [
        os.path.join(image_folder, img)
        for img in os.listdir(image_folder)
        if img.endswith((".jpg", ".png", ".jpeg"))
    ]

    for image_path in tqdm(image_files):
        if image_path in embeddings:
            continue
        try:
            image = load_image(image_path)
            image_embedding = image_model.encode(
                image, convert_to_tensor=True
            )  # <- Näin
            embeddings[image_path] = image_embedding.detach().cpu().numpy()
        except Exception as e:
            print(f"Error embedding image {image_path}: {e}")

    # Save embeddings to pickle file
    with open(pickle_file, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings


def search_similar(query_embedding, embeddings, top_k=5):
    image_paths = list(embeddings.keys())
    emb_matrix = np.array(list(embeddings.values()))

    similarities = cosine_similarity([query_embedding], emb_matrix)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [(image_paths[i], similarities[i]) for i in top_indices]


def visualize_images(image_paths, title="Image Results"):
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
    if len(image_paths) == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, img_path in zip(axes, image_paths):
        img = Image.open(img_path[0]) if isinstance(img_path[0], str) else img_path[0]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(str(img_path[1]))

    plt.suptitle(title)
    plt.show()


def upload_and_search(image_model, embeddings, device):
    uploader = widgets.FileUpload(accept="image/*", multiple=False)
    search_button = widgets.Button(description="Search Similar Images")
    output = widgets.Output()

    display(uploader, search_button, output)

    def on_search_clicked(b):
        with output:
            clear_output(wait=True)
            if uploader.value:
                print(uploader.value)
                uploaded_file = list(uploader.value)[0]
                image = Image.open(io.BytesIO(uploaded_file["content"])).convert("RGB")
                visualize_images([(image, 0)])
                image_embedding = (
                    image_model.encode(image, convert_to_tensor=True)
                    .to(device)
                    .detach()
                    .cpu()
                    .numpy()
                )

                results = search_similar(image_embedding, embeddings)
                print("Top similar images:", results)
                visualize_images(results, title="Top Similar Images")
            else:
                print("Please upload an image before searching.")

    search_button.on_click(on_search_clicked)


def search_with_image(image_model, embeddings, device, image_path):
    sample_image = load_image(image_path)
    visualize_images([(sample_image, 0)])
    image_query_embedding = (
        image_model.encode(sample_image, convert_to_tensor=True)
        .to(device)
        .detach()
        .cpu()
        .numpy()
    )

    results = search_similar(image_query_embedding, embeddings)

    visualize_images(results)


def search_with_text(text_model, text, embeddings, device):
    text_query_embedding = (
        text_model.encode(text, convert_to_tensor=True)
        .to(device)
        .detach()
        .cpu()
        .numpy()
    )

    results = search_similar(text_query_embedding, embeddings)

    visualize_images(results, title=f"Text search: {text}")


def create_azure_search_index(embeddings, index_name):
    # Azure Search configuration
    service_name = os.environ["AZURE_SEARCH_SERVICE_NAME"]
    admin_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    endpoint = f"https://{service_name}.search.windows.net"

    # Define the index schema as a raw JSON payload
    index_payload = {
        "name": index_name,
        "fields": [
            {
                "name": "id",
                "type": "Edm.String",
                "key": True,
                "searchable": False,
                "filterable": False,
                "sortable": False,
                "facetable": False,
            },
            {
                "name": "image_path",
                "type": "Edm.String",
                "searchable": False,
                "filterable": False,
                "sortable": False,
                "facetable": False,
            },
            {
                "name": "embedding",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "filterable": False,
                "sortable": False,
                "facetable": False,
                "retrievable": True,
                "dimensions": list(embeddings.values())[0].shape[0],
                "vectorSearchConfiguration": "myHnsw",
            },
        ],
        "vectorSearch": {
            "algorithmConfigurations": [
                {
                    "name": "myHnsw",
                    "kind": "hnsw",
                    "hnswParameters": {
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine",
                    },
                }
            ]
        },
    }

    # Make REST API call to create index
    headers = {"Content-Type": "application/json", "api-key": admin_key}

    create_index_url = f"{endpoint}/indexes/{index_name}?api-version=2023-07-01-Preview"

    response = requests.put(create_index_url, headers=headers, json=index_payload)

    if response.status_code == 201:
        print(f"Successfully created index {index_name}")
    else:
        print(f"Failed to create index. Status code: {response.status_code}")
        print(f"Error message: {response.text}")


def upload_embeddings_to_azure_search(embeddings, index_name):
    # Upload embeddings to Azure Search

    # Azure Search configuration
    service_name = os.environ["AZURE_SEARCH_SERVICE_NAME"]
    admin_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    endpoint = f"https://{service_name}.search.windows.net"

    # Create search client
    credential = AzureKeyCredential(admin_key)
    client = SearchClient(
        endpoint=endpoint, index_name=index_name, credential=credential
    )
    # Create search index if it doesn't exist
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

    if index_name not in [index.name for index in index_client.list_indexes()]:
        raise ValueError(f"Index {index_name} does not exist")

    # Prepare documents for upload
    docs = []
    for image_path, embedding in embeddings.items():
        doc = {
            "id": image_path.split("/")[-1].split(".")[0],
            "image_path": image_path,
            "embedding": embedding.tolist(),  # Convert numpy array to list
        }
        docs.append(doc)

    # Upload in batches of 1000 documents
    batch_size = 1000
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        try:
            results = client.upload_documents(documents=batch)
            print(f"Uploaded batch {i//batch_size + 1}")
        except Exception as e:
            print(f"Failed to upload batch {i//batch_size + 1}: {e}")


def get_embeddings_from_azure_search(index_name):
    # Azure Search configuration
    service_name = os.environ["AZURE_SEARCH_SERVICE_NAME"]
    admin_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    endpoint = f"https://{service_name}.search.windows.net"

    # Initialize the search client
    credential = AzureKeyCredential(admin_key)
    client = SearchClient(
        endpoint=endpoint, index_name=index_name, credential=credential
    )

    # Get all documents from the index
    results = list(client.search("*", select=["id", "image_path", "embedding"]))

    # Create embeddings dictionary
    embeddings = {}
    for doc in results:
        image_path = doc["image_path"]
        embedding = np.array(doc["embedding"])  # Convert list back to numpy array
        embeddings[image_path] = embedding

    return embeddings
