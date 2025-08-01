import asyncio
import os

from douyin_scraper.douyin.web.web_crawler import DouyinWebCrawler
import polars as pl
from tqdm import tqdm

async def main():
    related_path = './data/douyin_related_videos.parquet.zstd'
    if os.path.exists(related_path):
        video_df = pl.read_parquet(related_path)
    
    user_posts_path = './data/douyin_videos.parquet.zstd'
    if os.path.exists(user_posts_path):
        user_video_df = pl.read_parquet(user_posts_path)
        video_df = pl.concat([user_video_df, video_df], how='diagonal_relaxed').unique(subset=['aweme_id'])

    print(f"Starting with {video_df.shape[0]} videos.")

    crawler = DouyinWebCrawler()

    all_results = []
    for video in tqdm(video_df.sample(500).to_dicts()):
        video_id = video['aweme_id']
        try:
            result = await crawler.fetch_related_videos(video_id)
            all_results.extend(result['aweme_list'])
        except Exception as e:
            print(f"Error fetching related videos for {video_id}: {e}")

    schema_overrides = {
        'duet_origin_item': pl.Struct({
            'feed_comment_config': pl.Struct({'dummy': pl.Null}),
            'show_follow_button': pl.Struct({'dummy': pl.Null}),
            'poi_biz': pl.Struct({'dummy': pl.Null}),
        })
    }

    df = pl.concat([video_df, pl.from_dicts(all_results, infer_schema_length=len(all_results))], how='diagonal_relaxed')\
        .unique(subset=['aweme_id'])
    
    print(f"Total videos after merging: {df.shape[0]}")
        
    if 'duet_origin_item' in df.columns:
        df = df.drop('duet_origin_item')

    if 'show_follow_button' in df.columns:
        df = df.drop('show_follow_button')

    if 'entertainment_product_info' in df.columns:
        df = df.drop('entertainment_product_info')

    df.write_parquet(related_path, compression='zstd')

if __name__ == "__main__":
    asyncio.run(main())