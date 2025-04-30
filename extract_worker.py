import json
import re
from collections import defaultdict

folder = "logs/2025-03-26_0215_zero_drop"

worker_results = defaultdict(set)  # epoch -> list of accuracies

# Process each worker's log file
for worker_id in range(3):
    fn = f"worker_dynamic_bound_loss_log{worker_id}.txt"
    with open(f"{folder}/{fn}") as f:
        for line in f:
            match = re.search(r'{.*?}', line)
            if match and "'eval_accuracy':" in line and "'epoch':" in line:
                data_str = match.group(0).replace("'", '"')
                try:
                    data = json.loads(data_str)
                    worker_results[data['epoch']].add(data['eval_accuracy'])
                except json.JSONDecodeError:
                    continue

# Calculate averages and store results
results = []
for epoch, accuracies in worker_results.items():
    avg_accuracy = sum(accuracies) / len(accuracies)
    results.append({
        'epoch': epoch,
        'avg_accuracy': round(avg_accuracy, 4),
        'worker_accuracies': [round(acc, 4) for acc in accuracies]
    })

# Sort results by epoch number
results.sort(key=lambda x: x['epoch'])

# Print results in a readable format
for result in results:
    print(f"Epoch {result['epoch']}: Avg Test Accuracy={result['avg_accuracy']}, individual={result['worker_accuracies']}")