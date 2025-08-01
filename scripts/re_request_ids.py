import asyncio
import datetime
import os
from urllib.parse import urlencode

import polars as pl
import requests
from tqdm import tqdm

from douyin_scraper.douyin.web.web_crawler import DouyinWebCrawler
from douyin_scraper.base_crawler import BaseCrawler
from douyin_scraper.douyin.web.endpoints import DouyinAPIEndpoints
from douyin_scraper.douyin.web.models import PostDetail
from douyin_scraper.douyin.web.utils import BogusManager

async def main():
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
    sections = section_df.head(1)['section_bits'].to_list()

    crawler = DouyinWebCrawler()

    # result = await crawler.fetch_one_video(video_df['aweme_id'].to_list()[0])

    start_time = datetime.datetime(2025, 6, 1, 10, 0, 0)
    current_time = start_time
    milliseconds = 0

    kwargs = await crawler.get_douyin_headers()
    # 创建一个基础爬虫
    base_crawler = BaseCrawler(proxies=kwargs["proxies"], crawler_headers=kwargs["headers"])
    # 生成一个作品详情的带有a_bogus加密参数的Endpoint
    # 创建一个作品详情的BaseModel参数
    params = PostDetail(aweme_id='')
    params_dict = params.dict()
    params_dict["msToken"] = ''
    a_bogus = BogusManager.ab_model_2_endpoint(params_dict, kwargs["headers"]["User-Agent"])

    pbar = tqdm()
    all_results = []
    for video in video_df.to_dicts():
        aweme_id = video['aweme_id']

        pbar.update(1)

        try:
            async with base_crawler as crawler:
                # 生成一个作品详情的带有加密参数的Endpoint
                # 2024年6月12日22:41:44 由于XBogus加密已经失效，所以不再使用XBogus加密参数，转移至a_bogus加密参数。
                # endpoint = BogusManager.xb_model_2_endpoint(
                #     DouyinAPIEndpoints.POST_DETAIL, params.dict(), kwargs["headers"]["User-Agent"]
                # )
                params_dict['aweme_id'] = aweme_id
                endpoint = f"{DouyinAPIEndpoints.POST_DETAIL}?{urlencode(params_dict)}&a_bogus={a_bogus}"

                response = await crawler.fetch_get_json(endpoint)
                all_results.append({
                    'aweme_id': aweme_id,
                    'result': response
                })
        except Exception as e:
            print(f"Error fetching data for video ID {aweme_id}: {e}")

if __name__ == "__main__":
    asyncio.run(main())