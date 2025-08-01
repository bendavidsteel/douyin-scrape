import matplotlib.pyplot as plt
import polars as pl
from datetime import datetime

def main():
    video_df = pl.read_parquet('./data/douyin_related_videos.parquet.zstd', columns=['aweme_id'])
    video_df = video_df.with_columns([
        pl.col('aweme_id').cast(pl.UInt64).map_elements(lambda i: format(i, '064b'), pl.String).alias('aweme_id_bits')
    ]).with_columns([
        pl.from_epoch(pl.col('aweme_id_bits').str.slice(0, 32).map_elements(lambda x: int(x, 2), pl.Int64)).alias('timestamp'),
        pl.col('aweme_id_bits').str.slice(32, 10).map_elements(lambda x: int(x, 2), pl.Int64).alias('millisecond'),
        pl.col('aweme_id_bits').str.slice(42, 22).alias('section_bits')
    ])
    
    # Extract temporal components from timestamps
    temporal_df = video_df.with_columns([
        pl.col('timestamp').dt.hour().alias('hour_of_day'),
        pl.col('timestamp').dt.minute().alias('minute_of_hour'),
        pl.col('timestamp').dt.second().alias('second_of_minute')
    ])
    
    # Hour distribution (0-23)
    hourly_data = temporal_df.group_by('hour_of_day').agg(pl.count().alias('count'))
    complete_hours = pl.DataFrame({
        'hour_of_day': list(range(24)),
        'count': [0] * 24
    }).join(hourly_data, on='hour_of_day', how='left', suffix='_actual').with_columns([
        pl.coalesce([pl.col('count_actual'), pl.col('count')]).alias('count')
    ]).select(['hour_of_day', 'count']).sort('hour_of_day')
    
    # Minute distribution (0-59)
    minute_data = temporal_df.group_by('minute_of_hour').agg(pl.count().alias('count'))
    complete_minutes = pl.DataFrame({
        'minute_of_hour': list(range(60)),
        'count': [0] * 60
    }).join(minute_data, on='minute_of_hour', how='left', suffix='_actual').with_columns([
        pl.coalesce([pl.col('count_actual'), pl.col('count')]).alias('count')
    ]).select(['minute_of_hour', 'count']).sort('minute_of_hour')
    
    # Second distribution (0-59)
    second_data = temporal_df.group_by('second_of_minute').agg(pl.count().alias('count'))
    complete_seconds = pl.DataFrame({
        'second_of_minute': list(range(60)),
        'count': [0] * 60
    }).join(second_data, on='second_of_minute', how='left', suffix='_actual').with_columns([
        pl.coalesce([pl.col('count_actual'), pl.col('count')]).alias('count')
    ]).select(['second_of_minute', 'count']).sort('second_of_minute')
    
    # Create subplots for temporal and bit distributions
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Hour distribution
    ax1.bar(complete_hours['hour_of_day'], complete_hours['count'], color='skyblue', edgecolor='navy')
    ax1.set_title("Video Frequency by Hour of Day")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Number of Videos")
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(axis='y', alpha=0.3)
    
    # Minute distribution
    ax2.bar(complete_minutes['minute_of_hour'], complete_minutes['count'], color='lightcoral', edgecolor='darkred')
    ax2.set_title("Video Frequency by Minute of Hour")
    ax2.set_xlabel("Minute of Hour")
    ax2.set_ylabel("Number of Videos")
    ax2.set_xticks(range(0, 60, 10))
    ax2.grid(axis='y', alpha=0.3)
    
    # Second distribution
    ax3.bar(complete_seconds['second_of_minute'], complete_seconds['count'], color='lightgreen', edgecolor='darkgreen')
    ax3.set_title("Video Frequency by Second of Minute")
    ax3.set_xlabel("Second of Minute")
    ax3.set_ylabel("Number of Videos")
    ax3.set_xticks(range(0, 60, 10))
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot bits 32 to 42 distribution (fixed)
    bits_32_42 = video_df.group_by('millisecond').agg(pl.count().alias('count'))
    
    # Create complete range for 10 bits (0-1023) and fill missing values with 0
    complete_milliseconds = pl.DataFrame({
        'millisecond': list(range(1024)),
        'count': [0] * 1024
    }).join(bits_32_42, on='millisecond', how='left', suffix='_actual').with_columns([
        pl.coalesce([pl.col('count_actual'), pl.col('count')]).alias('count')
    ]).select(['millisecond', 'count']).sort('millisecond')
    
    ax4.plot(complete_milliseconds['millisecond'], complete_milliseconds['count'], 
             marker='o', linestyle='-', color='purple', markersize=1)
    ax4.set_title("Distribution of Bits 32-42 (Millisecond Values)")
    ax4.set_xlabel("Bit Value (0-1023)")
    ax4.set_ylabel("Frequency")
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('./figs/comprehensive_video_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print temporal statistics
    print("=== TEMPORAL POSTING PATTERNS ===")
    print(f"Peak posting hour: {complete_hours.filter(pl.col('count') == pl.col('count').max())['hour_of_day'][0]}:00")
    print(f"Peak posting minute: {complete_minutes.filter(pl.col('count') == pl.col('count').max())['minute_of_hour'][0]} minutes past the hour")
    print(f"Peak posting second: {complete_seconds.filter(pl.col('count') == pl.col('count').max())['second_of_minute'][0]} seconds past the minute")
    
    # Show some interesting statistics
    total_videos = len(video_df)
    print(f"\nTotal videos analyzed: {total_videos:,}")
    
    # Find quiet vs busy periods
    busy_hours = complete_hours.filter(pl.col('count') > pl.col('count').mean())
    quiet_hours = complete_hours.filter(pl.col('count') < pl.col('count').mean() * 0.5)
    
    print(f"Busiest hours: {busy_hours['hour_of_day'].to_list()}")
    print(f"Quietest hours: {quiet_hours['hour_of_day'].to_list()}")
    
    # Check for second-level patterns (might indicate bot activity)
    second_variance = complete_seconds['count'].var()
    second_mean = complete_seconds['count'].mean()
    print(f"\nSecond-level posting variance: {second_variance:.2f}")
    print(f"Second-level posting mean: {second_mean:.2f}")
    if second_variance / second_mean > 2:
        print("High variance in second-level posting detected - possible automated posting patterns")
    else:
        print("Relatively uniform second-level posting - appears more natural")
    
    # Count unique values in bits 42 to 64 (fixed)
    unique_counts = video_df.group_by('section_bits').agg(pl.count().alias('count')).sort('count', descending=True)
    
    print("Unique counts for bits 42 to 64:")
    print(unique_counts.head(10))  # Show top 10 most frequent
    print(f"\nTotal unique section_bits: {len(unique_counts)}")
    print(f"Total videos: {len(video_df)}")

if __name__ == "__main__":
    main()