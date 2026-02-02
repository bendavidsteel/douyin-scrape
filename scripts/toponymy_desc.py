import glob
from typing import List

import polars as pl
import toponymy
import toponymy.embedding_wrappers
import toponymy.llm_wrappers
from tqdm import tqdm
import parampacmap
import vllm

from scrape_related_posts import PARTITIONS_DIR

class VLLMTranslator(toponymy.llm_wrappers.AsyncVLLM):
    def translate_text(self, texts: List[str]) -> List[str]:
        prompts = [f"Translate the following text to English: {text}" for text in texts]
        messages = [[{"role": "user", "content": prompt + self.extra_prompting}] for prompt in prompts]
        
        sampling_params = vllm.SamplingParams(
            temperature=0.7, 
            max_tokens=128, 
            repetition_penalty=1.2,
        )
        
        chat_kwargs = {
            'messages': messages,
            'sampling_params': sampling_params,
            'chat_template_kwargs': {'enable_thinking': False},
            'use_tqdm': True
        }

        try:
            outputs = self.llm.chat(**chat_kwargs)
        except vllm.v1.engine.exceptions.EngineDeadError:
            self._start_engine()  # Restart the engine if it fails
            outputs = self.llm.chat(**chat_kwargs)

        return [output.outputs[0].text for output in outputs]

def main():
    partition_files = list(PARTITIONS_DIR.glob('*.parquet.zstd'))
    df = pl.DataFrame()
    for file_path in tqdm(partition_files):
        file_df = pl.read_parquet(file_path, columns=['aweme_id', 'create_time', 'author', 'caption', 'desc', 'statistics', 'region'])
        df = pl.concat([df, file_df], how='diagonal_relaxed')

    print(f"Number of posts: {df.shape[0]}")

    texts = df['desc'].to_list()

    embedding_model_name = 'intfloat/multilingual-e5-small'

    max_model_len = 512
    embedding_model = toponymy.embedding_wrappers.VLLMEmbedder(embedding_model_name, kwargs={'gpu_memory_utilization': 0.1, 'max_model_len': max_model_len})

    short_text = df['desc'].str.slice(0, max_model_len).to_list()
    embeddings = embedding_model.encode(short_text, show_progress_bar=True)

    umap_model = parampacmap.ParamPaCMAP(
        n_components=2,
        verbose=True
    )
    umap_vectors = umap_model.fit_transform(embeddings)

    llm_model_name = 'Qwen/Qwen3-4B'
    llm = VLLMTranslator(llm_model_name, gpu_memory_utilization=0.7, max_model_len=8192)

    translated_texts = llm.translate_text(texts)

    clusterer = toponymy.ToponymyClusterer(min_clusters=4, verbose=True, base_min_cluster_size=100)
    clusterer.fit(clusterable_vectors=umap_vectors, embedding_vectors=embeddings)

    topic_model = toponymy.Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedding_model,
        clusterer=clusterer,
        object_description="Douyin post descriptions",
        corpus_description="Douyin post dataset",
    )

    # Note on data types for fit() method:
    # - text: Python list of strings (not numpy array)
    # - document_vectors: numpy array of shape (n_documents, embedding_dimension)
    # - document_map: numpy array of shape (n_documents, clustering_dimension)
    topic_model.fit(translated_texts, embeddings, umap_vectors)

    topic_names = topic_model.topic_names_
    topics_per_document = [cluster_layer.topic_name_vector for cluster_layer in topic_model.cluster_layers_]

    topic_df = df.with_columns([pl.Series(name=f"cluster_layer_{i}", values=c.topic_name_vector) for i, c in enumerate(topic_model.cluster_layers_)])\
        .with_columns([
            pl.Series(name='umap_vector', values=umap_vectors),
            pl.Series(name='translated_desc', values=translated_texts),
        ])

    topic_df.write_parquet('./data/douyin_post_topics.parquet.zstd')


if __name__ == '__main__':
    main()