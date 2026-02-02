import asyncio
import os
from pathlib import Path
from douyin_scraper.douyin.web.web_crawler import DouyinWebCrawler
import polars as pl
from tqdm import tqdm
import random

# Configuration
MAX_WORKERS = 8  # Adjust this value based on your needs and rate limits
DATA_DIR = Path('./data')
PARTITIONS_DIR = DATA_DIR / 'related_partitions'
PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)

def get_partition_key(aweme_id):
    """Get partition key from aweme_id (first 8 characters)"""
    return str(aweme_id)[:4]

def get_partition_path(partition_key):
    """Get the file path for a partition"""
    return PARTITIONS_DIR / f'partition_{partition_key}.parquet.zstd'

def save_partition(df, partition_key):
    """Save a dataframe partition to disk"""
    if df.is_empty():
        return
    
    partition_path = get_partition_path(partition_key)
    
    # If partition exists, merge with existing data
    if partition_path.exists():
        existing_df = pl.read_parquet(partition_path)
        df = pl.concat([existing_df, df], how='diagonal_relaxed').unique(subset=['aweme_id'])
    
    df.write_parquet(partition_path, compression='zstd')

def partition_dataframe(df):
    """Partition a dataframe by aweme_id prefix and save each partition"""
    # Add partition key column
    df = df.with_columns(
        pl.col('aweme_id').cast(pl.Utf8).str.slice(0, 4).alias('partition_key')
    )
    
    # Group by partition key and save each group
    for partition_key, group_df in df.group_by('partition_key'):
        partition_key = partition_key[0]  # Extract from tuple
        group_df = group_df.drop('partition_key')
        save_partition(group_df, partition_key)

def load_sample_across_partitions(sample_size, keywords=None):
    """Load a sample of videos across all partitions"""
    partition_files = list(PARTITIONS_DIR.glob('*.parquet.zstd'))
    
    if not partition_files:
        return pl.DataFrame()
    
    # Calculate samples per partition
    samples_per_partition = max(1, sample_size // len(partition_files))
    remainder = sample_size % len(partition_files)
    
    sampled_dfs = []
    
    for i, partition_file in tqdm(enumerate(partition_files), desc="Sampling partitions", total=len(partition_files)):
        # Add extra sample for remainder distribution
        current_sample_size = samples_per_partition + (1 if i < remainder else 0)
        
        # Load partition
        partition_df = pl.scan_parquet(partition_file)
        
        if keywords and 'desc' in partition_df.columns:
            # Filter by keywords if provided
            filtered_df = partition_df.filter(pl.col('desc').str.contains_any(keywords))
            filtered_df = filtered_df.collect()
            if not filtered_df.is_empty():
                sample_count = min(current_sample_size, filtered_df.height)
                if sample_count > 0:
                    sampled_dfs.append(filtered_df.sample(sample_count))
        else:
            partition_df = partition_df.collect()
            # Random sample
            sample_count = min(current_sample_size, partition_df.height)
            if sample_count > 0:
                sampled_dfs.append(partition_df.sample(sample_count))
    
    if sampled_dfs:
        return pl.concat(sampled_dfs, how='diagonal_relaxed')
    return pl.DataFrame()

def get_total_video_count():
    """Get total count of videos across all partitions"""
    total = 0
    for partition_file in PARTITIONS_DIR.glob('*.parquet.zstd'):
        df = pl.scan_parquet(partition_file).select(pl.count()).collect()
        total += df['count'][0]
    return total

async def fetch_video_worker(semaphore, crawler, video):
    """Worker function to fetch related videos for a single video"""
    async with semaphore:
        video_id = video['aweme_id']
        try:
            result = await crawler.fetch_related_videos(video_id)
            return result.get('aweme_list', [])
        except Exception as e:
            print(f"Error fetching related videos for {video_id}: {e}")
            return []

async def process_videos_parallel(sample_df, max_workers=MAX_WORKERS):
    """Process videos in parallel with limited workers"""
    crawler = DouyinWebCrawler()
    semaphore = asyncio.Semaphore(max_workers)
    
    # Create tasks for all videos
    tasks = [
        fetch_video_worker(semaphore, crawler, video)
        for video in sample_df.to_dicts()
    ]
    
    # Process tasks with progress bar
    all_results = []
    with tqdm(total=len(tasks), desc="Fetching related videos") as pbar:
        for completed_task in asyncio.as_completed(tasks):
            video_results = await completed_task
            pbar.update(1)
            if not video_results:
                continue
            all_results.extend(video_results)
    
    return all_results

def process_and_save_results(results):
    """Process results and save to appropriate partitions"""
    if not results:
        return
    
    # Convert results to dataframe
    df = pl.from_dicts(results, infer_schema_length=len(results))
    
    # Clean up problematic columns
    columns_to_drop = ['duet_origin_item', 'show_follow_button', 'entertainment_product_info']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col)
    
    # Partition and save
    partition_dataframe(df)

async def main():
    
    while True:
        # Load keywords
        all_keywords = []
        with open('./data/keywords.txt', 'r') as f:
            keywords_lines = f.readlines()
            for line in keywords_lines:
                keywords, explanation = line.split('(')
                keywords = keywords.split('or')
                keywords = [k.strip() for k in keywords]
                all_keywords.extend(keywords)
        
        # Get total video count
        total_videos = get_total_video_count()
        print(f"Starting with {total_videos} videos across partitions.")
        
        # Sample videos
        total_sample = 4000
        
        # Sample interesting videos (with keywords)
        interesting_sample_df = load_sample_across_partitions(
            int(0.5 * total_sample), 
            keywords=all_keywords
        )
        
        # Sample general videos (without keywords)
        general_sample_df = load_sample_across_partitions(
            int(0.5 * total_sample)
        )
        
        # Combine samples
        sample_df = pl.concat([interesting_sample_df, general_sample_df], how='diagonal_relaxed')
        
        # Remove duplicates in the sample
        sample_df = sample_df.unique(subset=['aweme_id'])
        
        # Process videos in parallel
        print(f"Processing {sample_df.shape[0]} videos with {MAX_WORKERS} workers...")
        all_results = await process_videos_parallel(sample_df, MAX_WORKERS)
        
        # Save results to partitions
        print(f"Saving {len(all_results)} new videos to partitions...")
        process_and_save_results(all_results)
        
        # Print final statistics
        final_total = get_total_video_count()
        print(f"Total videos after merging: {final_total}")
        print(f"New videos added: {final_total - total_videos}")
        
        # Create a backup snapshot (optional - this will use disk space)
        # You might want to do this less frequently
        backup_dir = DATA_DIR / 'backups'
        backup_dir.mkdir(parents=True, exist_ok=True)
        for partition_file in PARTITIONS_DIR.glob('*.parquet.zstd'):
            import shutil
            shutil.copy2(partition_file, backup_dir / partition_file.name)

if __name__ == "__main__":
    asyncio.run(main())