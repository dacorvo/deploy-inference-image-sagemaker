import argparse
import csv
from pathlib import Path

METRICS_NAMES = ["encoding_time", "total_time", "decoding_time"]


def read_locust_csv_stats(filepath: str | Path):
    with open(filepath, newline='') as csvfile:
        stats_reader = csv.reader(csvfile, delimiter=',')
        labels = next(stats_reader)
        metrics = {}
        for row in stats_reader:
            metric = dict(zip(labels, row))
            if metric["Name"] in METRICS_NAMES:
                metrics[metric["Name"]] = metric
        return metrics


def summarize(metrics: dict[str, str]):
    """Summarize the metrics extracted from a Locust CSV stats file

    Args:
        metrics: the Locust metrics
    Returns:
        A tuple of prompt_tokens, generated_tokens, request-per-second, time-to-first-token, throughput
    """
    for name in METRICS_NAMES:
        assert name in metrics
    total_time = float(metrics["total_time"]["Average Response Time"])
    encoding_time = float(metrics["encoding_time"]["Average Response Time"])
    prompt_tokens = float(metrics["encoding_time"]["Average Content Size"])
    decoding_time = float(metrics["decoding_time"]["Average Response Time"])
    generated_tokens = float(metrics["decoding_time"]["Average Content Size"])
    rps = 1000 / total_time
    ttft = encoding_time / 1000
    throughput = 1000 * generated_tokens / decoding_time
    return prompt_tokens, generated_tokens, rps, ttft, throughput


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="*", type=str, default=".")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--summary_file", type=str, default="benchmark_summary.csv")
    args = parser.parse_args()
    with open(args.summary_file, 'w') as summary_file:
        summary_writer = csv.writer(summary_file, delimiter=',')
        summary_writer.writerow([
            "Run Name",
            "Average prompt tokens",
            "Average generated tokens",
            "Requests per Second",
            "Time-to-first-token (s)",
            "Output Token Throughput (t/s)"
        ])
        csv_stats_path = Path(args.directory)
        csv_stats_files = csv_stats_path.glob(f"{args.prefix}*.csv_stats.csv")
        for csv_stat_file in csv_stats_files:
            metrics = read_locust_csv_stats(csv_stat_file)
            summary = summarize(metrics)
            # Get the filename without the extension
            csv_stat_name = csv_stat_file.name.split('.')[0]
            # Extract the run name
            run_name = csv_stat_name.removeprefix(args.prefix)
            summary_writer.writerow((run_name,) + summary)
