from ultralytics.utils.benchmarks import benchmark

benchmark(model="./models/pose.pt", data="../datasets/hand-keypoints/data.yaml", half=True)

