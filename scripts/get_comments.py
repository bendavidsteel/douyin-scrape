import asyncio
import os

from douyin_scraper.douyin.web.web_crawler import DouyinWebCrawler
import polars as pl
from tqdm import tqdm

from scrape_related_posts import load_sample_across_partitions

async def fetch_comments_worker(semaphore, crawler, video_id):
    async with semaphore:
        has_more = 1
        cursor = 0
        all_comments = []
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
        return all_comments
    
async def fetch_comments_parallel(video_df, max_workers=5):
    crawler = DouyinWebCrawler()
    semaphore = asyncio.Semaphore(max_workers)
    
    tasks = [
        fetch_comments_worker(semaphore, crawler, video['aweme_id'])
        for video in video_df.to_dicts()
    ]
    
    all_comments = []
    with tqdm(total=len(tasks), desc="Fetching video comments") as pbar:
        for completed_task in asyncio.as_completed(tasks):
            comments = await completed_task
            pbar.update(1)
            if comments:
                all_comments.extend(comments)
    
    return all_comments

async def main():
    sample_size = 1000
    while True:
        comment_path = './data/douyin_comments.parquet.zstd'
        comment_df = pl.read_parquet(comment_path)
        
        sample_df = load_sample_across_partitions(sample_size)

        sample_df = sample_df.filter(~pl.col('aweme_id').is_in(comment_df.select('aweme_id').unique()['aweme_id'].to_list()))

        all_comments = await fetch_comments_parallel(sample_df, max_workers=1)

        comment_df = pl.concat([comment_df, pl.from_dicts(all_comments, infer_schema_length=len(all_comments))], how='diagonal_relaxed')

        print(f"Total comments after merging: {comment_df.shape[0]}")

        comment_df.write_parquet(comment_path, compression='zstd')

if __name__ == "__main__":
    asyncio.run(main())