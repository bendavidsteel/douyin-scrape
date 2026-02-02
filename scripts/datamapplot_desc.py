


import datamapplot
import pandas as pd
import polars as pl
from tqdm import tqdm

from scrape_related_posts import PARTITIONS_DIR

def main():
    topic_df = pl.read_parquet('./data/douyin_post_topics.parquet.zstd')

    title = 'Douyin Post Descriptions'

    topic_cols = [col for col in topic_df.columns if col.startswith('cluster_layer_')]

    max_desc_len = 50

    topic_df = topic_df.with_columns(pl.from_epoch(pl.col('create_time')))\
        .with_columns(
            (10 + pl.col('statistics').struct.field('digg_count')).log1p().alias('diggCountLog1p'),
            pl.concat_str([
                # pl.format("Desc: {}", pl.col('desc')),
                pl.when(pl.col('translated_desc').str.len_chars() > max_desc_len).then(pl.format("{}...", pl.col('translated_desc').str.slice(0, max_desc_len))).otherwise(pl.col('translated_desc')),
                # pl.format("Region: {}", pl.when(pl.col('region').is_null()).then(pl.lit("N/A")).otherwise(pl.col('region'))),
                # pl.format("Create Time: {}", pl.col('create_time')),
                # pl.format("Like Count: {}", pl.col('statistics').struct.field('digg_count')),
            ], separator='\n').alias('hover_text')
        )

    plot = datamapplot.create_interactive_plot(
        topic_df['umap_vector'].to_numpy(),
        *[topic_df[col].to_numpy() for col in topic_cols],
        hover_text=topic_df['hover_text'].to_numpy(),
        title=title,
        enable_search=True,
        darkmode=True,
        marker_size_array=topic_df['diggCountLog1p'].to_numpy(),
        font_family="Cinzel",
        minify_deps=True,
        histogram_data=topic_df['create_time'].to_numpy(),
        histogram_group_datetime_by='quarter',
        histogram_range=(pd.to_datetime('2025-01-01'), pd.to_datetime('2025-09-01')),
    )

    plot.save("./data/douyin_post_topics.html")

if __name__ == "__main__":
    main()