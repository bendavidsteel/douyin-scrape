import asyncio
import os

from douyin_scraper.douyin.web.web_crawler import DouyinWebCrawler
import polars as pl

async def main():
    video_path = 'douyin_videos.parquet.zstd'
    if os.path.exists(video_path):
        video_df = pl.read_parquet(video_path)
    else:
        video_df = pl.DataFrame()

    sec_uids = [
        'MS4wLjABAAAA4W5VX_Uamm8KpIhzBgZsfEi8rxCqK0L5Ph1T_RuBiwm9UnH7NJkTRRlJ_9EaM0GK',
        'MS4wLjABAAAA4vgRHGrSG6rPlffm3RvwHWL8TBq7O4YnM5jHUNXz0-s',
        'MS4wLjABAAAA-q30KOkHxTIpaBibao2fuoPgI7vRE1e4V2ySmJibmTyuR65lCCJM3lNNzAyZMrAs',
        'MS4wLjABAAAAJUrMay85Tgu4Fapq4E4jWP0EhmC16Jn4W1nZ5xEZE3Q',
        'MS4wLjABAAAA8B8Rkt2KE1obYtNeSdY7KAQXGgsRNIRDB5mlvhj-WxY'
    ]

    crawler = DouyinWebCrawler()

    all_results = []
    for sec_uid in sec_uids:
        has_more = True
        cursor = 0
        while has_more:
            results = await crawler.fetch_user_post_videos(sec_uid, cursor, 20)
            if 'aweme_list' not in results:
                break
            all_results.extend(results['aweme_list'])
            has_more = bool(results.get('has_more', False))
            # cursor = results.get('max_cursor', 0)
            cursor += 20

    schema_overrides = {
        'show_follow_button': pl.Struct({'dummy': pl.Null}),
        'video_game_data_channel_config': pl.Struct({'dummy': pl.Null}),
        'image_comment': pl.Struct({'dummy': pl.Null}),
        'series_basic_info': pl.Struct({'dummy': pl.Null}),
    }
    pl.concat([video_df, pl.from_dicts(all_results, infer_schema_length=len(all_results), schema_overrides=schema_overrides)], how='diagonal_relaxed')\
        .unique(subset=['aweme_id'])\
        .write_parquet(video_path, compression='zstd')

if __name__ == "__main__":
    asyncio.run(main())