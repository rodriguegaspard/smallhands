from ultralytics.utils.benchmarks import benchmark

benchmark(model="./models/smallhands.pt", data="../datasets/smallhands/hands.yaml", half=True)

