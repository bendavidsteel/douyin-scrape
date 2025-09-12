import asyncio
import os

from douyin_scraper.douyin.web.web_crawler import DouyinWebCrawler
import polars as pl
from tqdm import tqdm

async def main():
    comment_path = './data/douyin_comments.parquet.zstd'
    if os.path.exists(comment_path):
        comment_df = pl.read_parquet(comment_path)
    
    video_path = './data/douyin_videos.parquet.zstd'
    if os.path.exists(video_path):
        video_df = pl.read_parquet(video_path)

    crawler = DouyinWebCrawler()

    all_comments = []
    for video in tqdm(video_df.to_dicts()):
        video_id = video['aweme_id']
        has_more = 1
        cursor = 0
        try:
            while has_more:
                result = await crawler.fetch_video_comments(video_id, cursor)
                comments = result.get('comments', [])
                if not comments:
                    break
                all_comments.extend(comments)
                has_more = result.get('has_more', 0)
                cursor = result.get('cursor', len(result['comments']))
        except Exception as e:
            print(f"Error fetching video comments for {video_id}: {e}")

    comment_df = pl.concat([comment_df, pl.from_dicts(all_comments, infer_schema_length=len(all_comments))], how='diagonal_relaxed')

    print(f"Total comments after merging: {comment_df.shape[0]}")

    comment_df.write_parquet(comment_path, compression='zstd')

if __name__ == "__main__":
    asyncio.run(main())