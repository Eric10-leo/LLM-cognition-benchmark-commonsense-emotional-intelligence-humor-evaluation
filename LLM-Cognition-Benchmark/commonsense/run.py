import os
import json
import subprocess
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.dirname(__file__))
env = os.environ.copy()
env["PYTHONPATH"] = project_root

# ========== Step 1: Run all eval scripts ==========
eval_scripts = [
    "eval/eval_CommonsenseQA.py",
    "eval/eval_Social_IQA.py",
    "eval/eval_PIQA.py",
    "eval/eval_Commonsense-CN.py",
]

for script in eval_scripts:
    print(f"Running: {script}")
    subprocess.run(["python", script], check=True, env=env)

# ========== Step 2: Collect all acc results ==========
acc_results = []
results_root = "results"

for task_dir in os.listdir(results_root):
    task_path = os.path.join(results_root, task_dir)
    if os.path.isdir(task_path):
        for filename in os.listdir(task_path):
            if filename.startswith("acc_") and filename.endswith(".json"):
                with open(os.path.join(task_path, filename), "r", encoding="utf-8") as f:
                    result = json.load(f)
                    result["task"] = task_dir
                    acc_results.append(result)

# ========== Step 3: Per-task bar charts ==========
acc_by_task = {}
for item in acc_results:
    acc_by_task.setdefault(item["task"], []).append(item)

for task, entries in acc_by_task.items():
    entries.sort(key=lambda x: x["accuracy"], reverse=True)
    models = [e["model"] for e in entries]
    accs = [e["accuracy"] for e in entries]

    plt.figure(figsize=(10, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    bars = plt.bar(models, accs, color=colors[:len(models)])
    plt.title(f"Accuracy — {task}", fontsize=14)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.2%}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plot_path = os.path.join("results", f"acc_bar_{task.lower()}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")

# ========== Step 4: Cross-task comparison chart ==========
model_task_map = {}
for item in acc_results:
    model = item["model"]
    task = item["task"]
    acc = item["accuracy"]
    model_task_map.setdefault(model, {})[task] = acc

tasks = sorted({item["task"] for item in acc_results})
models = sorted(model_task_map.keys())

plt.figure(figsize=(12, 6))
bar_width = 0.15
x = range(len(models))
colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

for i, task in enumerate(tasks):
    accs = [model_task_map[model].get(task, 0) for model in models]
    plt.bar([pos + i * bar_width for pos in x], accs, width=bar_width, label=task, color=colors[i])

plt.xticks([pos + bar_width * 1.5 for pos in x], models, fontsize=11)
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Cross-Task Accuracy Comparison", fontsize=14)
plt.legend(title="Task", fontsize=10)
plt.tight_layout()
plt.savefig("results/acc_bar_cross_tasks.png", dpi=150)
plt.close()
print("Saved: results/acc_bar_cross_tasks.png")

# ========== Step 5: Markdown summary table ==========
markdown_path = "results/accuracy_summary.md"
with open(markdown_path, "w", encoding="utf-8") as f:
    f.write("| Task | Model | Accuracy |\n|------|--------|----------|\n")
    for item in sorted(acc_results, key=lambda x: (x["task"], -x["accuracy"])):
        f.write(f"| {item['task']} | {item['model']} | {item['accuracy']:.2%} |\n")

print(f"Markdown summary saved: {markdown_path}")
print("\nDone! All results generated.")
