"""Benchmark FlashSeg speed."""

from flashseg.analytics import Benchmark

bench = Benchmark(model_size="m", input_size=512, num_classes=21, device="cuda")
results = bench.run()

print(f"FPS: {results['fps']}")
print(f"Latency: {results['latency_ms']}ms")
print(f"Params: {results['params_m']}M")
print(f"Size: {results['size_mb']} MB")
