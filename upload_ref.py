import asyncio
from collections import OrderedDict
import os
import uuid
from openai import AsyncOpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()


openai_client = AsyncOpenAI()
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

embedding_model = "text-embedding-3-large"
collection_name = "openai_openapi"


qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=3072,
        distance=models.Distance.COSINE,
    ),
)


async def get_embeddings(texts, model=embedding_model):
    texts = list(map(lambda x: x.replace("\n", " "), texts))
    response = await openai_client.embeddings.create(input=texts, model=model)
    return list(map(lambda x: x.embedding, response.data))


async def generate_detailed_description(path_spec):
    prompt = (
        "Generate an exhaustive summary of the following OpenAPI path specification. Only include information from the specification, cover the basic idea and use cases of the endpoint. Write everything in one paragraph optimized for embedding creation.\n"
        "Do not respond with any additional comments or explanations, respond with the description only!\n"
        f"\n```yaml\n{path_spec}\n```"
    )

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0,
    )

    return response.choices[0].message.content


async def generate_all_descriptions(files_data):
    semaphore = asyncio.Semaphore(3)

    async def generate_description_for_file(file_data):
        async with semaphore:
            file_data["description"] = await generate_detailed_description(
                file_data["specification"]
            )

    tasks = [
        generate_description_for_file(file_data) for file_data in files_data.values()
    ]
    await asyncio.gather(*tasks)


async def generate_all_embeddings(files_data):
    embeddings = await get_embeddings(
        [file_data["description"] for file_data in files_data.values()]
    )
    for file_data, embedding in zip(files_data.values(), embeddings):
        file_data["embedding"] = embedding


def upload_all_embeddings(files_data):
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=file_data["embedding"],
            payload={
                "content": file_data["specification"],
                "description": file_data["description"],
            },
        )
        for file_data in files_data.values()
    ]

    batch_size = 3
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        qdrant_client.upload_points(collection_name=collection_name, points=batch)


async def main():
    directory_path = "paths"
    files_data = OrderedDict()

    print("Loading all files into memory...")
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".yaml"):
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                files_data[file_name] = {"specification": file.read()}

    print("Generating descriptions for all files...")
    await generate_all_descriptions(files_data)

    print("Generating embeddings for all descriptions...")
    await generate_all_embeddings(files_data)

    print("Uploading all embeddings to Qdrant...")
    upload_all_embeddings(files_data)

    print("All files have been processed and uploaded to Qdrant.")


# Run the main function
asyncio.run(main())
