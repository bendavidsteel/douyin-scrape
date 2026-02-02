import asyncio
import datetime
import os
import polars as pl
import requests
from tqdm import tqdm
from douyin_scraper.douyin.web.web_crawler import DouyinWebCrawler


class AsyncDouyinScraper:
    def __init__(self, num_workers=10, batch_size=10):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_dir = './data/'
        self.work_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
        self.processed_ids = set()
        self.save_lock = asyncio.Lock()
        self.pbar = tqdm()
        self.stop_workers = False
        self.current_dataframes = {}
        
    def get_dataframe_prefix(self, aweme_id):
        """Extract first 8 digits from aweme_id for dataframe file naming"""
        return aweme_id[:8]
    
    def get_dataframe_path(self, prefix):
        """Get the path for a specific dataframe file"""
        return f"{self.data_dir}{prefix}_sample_related_videos.parquet.ztsd"
    
    def load_dataframe(self, prefix):
        """Load a specific dataframe by prefix"""
        path = self.get_dataframe_path(prefix)
        if os.path.exists(path):
            return pl.read_parquet(path)
        else:
            return pl.DataFrame({'aweme_id': [], 'result': []})
    
    async def load_data(self):
        """Load existing sampled data and video data"""
        # Load processed IDs from all existing dataframe files
        self.processed_ids = set()
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_sample_related_videos.parquet.ztsd'):
                try:
                    df = pl.read_parquet(f"{self.data_dir}{filename}", columns=['aweme_id'])
                    self.processed_ids.update(df['aweme_id'].to_list())
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
            
        video_df = pl.read_parquet('./data/douyin_related_videos.parquet.zstd', columns=['aweme_id'])
        video_df = video_df.with_columns([
            pl.col('aweme_id').cast(pl.UInt64).map_elements(lambda i: format(i, '064b'), pl.String).alias('aweme_id_bits')
        ]).with_columns([
            pl.from_epoch(pl.col('aweme_id_bits').str.slice(0, 32).map_elements(lambda x: int(x, 2), pl.Int64)).alias('timestamp'),
            pl.col('aweme_id_bits').str.slice(32, 10).map_elements(lambda x: int(x, 2), pl.Int64).alias('millisecond'),
            pl.col('aweme_id_bits').str.slice(42, 22).alias('section_bits')
        ])
        
        section_df = video_df.group_by(pl.col('section_bits'))\
            .agg(pl.count().alias('count'))\
            .sort('count', descending=True)
        
        self.sections = section_df.head(1)['section_bits'].to_list()
        
    async def id_generator(self):
        """Generate aweme IDs to be processed"""
        start_time = datetime.datetime(2023, 6, 1, 10, 0, 0)
        current_time = start_time
        milliseconds = 0
        
        while not self.stop_workers:
            if milliseconds >= 1000:
                milliseconds = 0
                current_time += datetime.timedelta(seconds=1)
                
            timestamp_bits = format(int(current_time.timestamp()), '032b')
            milliseconds_bits = format(milliseconds, '010b')
            all_bits = timestamp_bits + milliseconds_bits + self.sections[0]
            aweme_id = str(int(all_bits, 2))
            
            if aweme_id not in self.processed_ids:
                await self.work_queue.put(aweme_id)
                self.processed_ids.add(aweme_id)
                
            milliseconds += 1
            
            # Prevent queue from growing too large
            if self.work_queue.qsize() > self.num_workers * 10:
                await asyncio.sleep(0.1)
                
    async def worker(self, worker_id):
        """Worker that fetches related videos"""
        crawler = DouyinWebCrawler()
        cols_to_remove = ['duet_origin_item', 'show_follow_button', 'entertainment_product_info']
        
        while not self.stop_workers:
            try:
                # Get work item with timeout to allow checking stop_workers
                aweme_id = await asyncio.wait_for(self.work_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
                
            self.pbar.update(1)
            
            try:
                response = await crawler.fetch_related_videos(aweme_id)
                
                # Clean response
                response['aweme_list'] = [
                    {k: v for k, v in item.items() if k not in cols_to_remove} 
                    for item in response['aweme_list']
                ]
                
                await self.results_queue.put({
                    'aweme_id': aweme_id,
                    'result': response
                })
                
            except Exception as e:
                print(f"Worker {worker_id} - Error fetching data for video ID {aweme_id}: {e}")
                
    async def result_processor(self):
        """Process and save results in batches"""
        all_results = []
        
        while not self.stop_workers or not self.results_queue.empty():
            try:
                # Get result with timeout
                result = await asyncio.wait_for(self.results_queue.get(), timeout=1.0)
                all_results.append(result)
            except asyncio.TimeoutError:
                pass
                
            # Save batch when threshold reached
            if len(all_results) >= self.batch_size:
                await self.save_batch(all_results)
                all_results = []
                
        # Save any remaining results
        if all_results:
            await self.save_batch(all_results)
            
    async def save_batch(self, results):
        """Save a batch of results to appropriate parquet files"""
        if not results:
            return
            
        async with self.save_lock:
            # Group results by prefix (first 8 digits of aweme_id)
            grouped_results = {}
            for result in results:
                prefix = self.get_dataframe_prefix(result['aweme_id'])
                if prefix not in grouped_results:
                    grouped_results[prefix] = []
                grouped_results[prefix].append(result)
            
            # Save each group to its corresponding file
            for prefix, group_results in grouped_results.items():
                # Load or get existing dataframe for this prefix
                if prefix not in self.current_dataframes:
                    self.current_dataframes[prefix] = self.load_dataframe(prefix)
                
                new_df = pl.from_dicts(group_results, infer_schema_length=len(group_results), strict=False)
                self.current_dataframes[prefix] = pl.concat([self.current_dataframes[prefix], new_df], how='diagonal_relaxed')
                
                # Save to file
                path = self.get_dataframe_path(prefix)
                self.current_dataframes[prefix].write_parquet(path, compression='zstd')

            # delete unused dataframes to free memory
            for prefix in list(self.current_dataframes.keys()):
                if prefix not in grouped_results:
                    del self.current_dataframes[prefix]
            
    async def run(self):
        """Main execution method"""
        await self.load_data()
        
        # Create tasks
        tasks = []
        
        # ID generator task
        tasks.append(asyncio.create_task(self.id_generator()))
        
        # Worker tasks
        for i in range(self.num_workers):
            tasks.append(asyncio.create_task(self.worker(i)))
            
        # Result processor task
        tasks.append(asyncio.create_task(self.result_processor()))
        
        try:
            # Run until interrupted
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            self.stop_workers = True
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            self.pbar.close()
            print("Scraping completed.")


async def main():
    # Adjust num_workers based on your API rate limits and system capabilities
    scraper = AsyncDouyinScraper(num_workers=8, batch_size=4096)
    await scraper.run()


if __name__ == "__main__":
    asyncio.run(main())