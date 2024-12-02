import itertools
import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# Utility function to evaluate recommendations
def evaluate_recommendations(ground_truth, response, metric="precision"):
    """
    Dummy evaluation function for recommendations.
    Replace with your actual metric evaluation logic.
    """
    # For simplicity, assume the metric is based on overlap
    ground_truth_set = set(ground_truth.split(", "))
    response_set = set(response.split(", "))
    if metric == "precision":
        return len(ground_truth_set & response_set) / len(response_set) if response_set else 0
    elif metric == "recall":
        return len(ground_truth_set & response_set) / len(ground_truth_set) if ground_truth_set else 0
    else:
        raise ValueError(f"Unsupported metric: {metric}")


# Step 1: Generate k samples and evaluate
def generate_and_evaluate_samples(model_id, dataset, k=5, metric="precision"):
    """
    Generate k samples for each prompt and evaluate their metrics.
    """
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    preferences = []
    for data in tqdm(dataset["train"], desc="Generating samples and evaluating metrics"):
        prompt = data["prompt"]
        ground_truth = data["completion"]

        # Generate k responses
        responses = []
        metrics = []
        for _ in range(k):
            output = pipe(prompt, max_new_tokens=256, num_return_sequences=1)
            response_text = output[0]["generated_text"]
            responses.append(response_text)

            # Evaluate the response
            metric_value = evaluate_recommendations(ground_truth, response_text, metric)
            metrics.append(metric_value)

        # Store results
        preferences.append({
            "prompt": prompt,
            "responses": responses,
            "metrics": metrics
        })

    return preferences


# Step 2: Generate pairwise preferences
def generate_pairwise_preferences(preferences, metric="precision"):
    """
    Generate pairwise preference dataset based on metric comparisons.
    """
    pairwise_data = []
    for item in tqdm(preferences, desc="Generating pairwise preferences"):
        prompt = item["prompt"]
        responses = item["responses"]
        metrics = item["metrics"]

        # Generate all combinations of responses (kC2)
        for (idx1, idx2) in itertools.combinations(range(len(responses)), 2):
            response_1 = responses[idx1]
            response_2 = responses[idx2]
            metric_1 = metrics[idx1]
            metric_2 = metrics[idx2]

            # Compare metrics and assign preference
            if metric_1 > metric_2:
                pairwise_data.append({
                    "prompt": prompt,
                    "response_1": response_1,
                    "response_2": response_2,
                    "preferred": "response_1"
                })
            elif metric_2 > metric_1:
                pairwise_data.append({
                    "prompt": prompt,
                    "response_1": response_1,
                    "response_2": response_2,
                    "preferred": "response_2"
                })

    return pairwise_data


# Main script
if __name__ == "__main__":
    # Load your dataset
    dataset = {
        "train": [
            {"prompt": "Recommend me some action movies.", "completion": "Mad Max, John Wick, Die Hard"},
            {"prompt": "Suggest me romantic comedies.", "completion": "Crazy Rich Asians, Notting Hill, Love Actually"}
        ]
    }

    # Parameters
    model_id = "gpt-3.5-turbo"
    k = 5
    metric = "precision"

    # Step 1: Generate samples and evaluate metrics
    print("Step 1: Generating and evaluating samples...")
    preferences = generate_and_evaluate_samples(model_id, dataset, k, metric)

    # Step 2: Generate pairwise preference dataset
    print("Step 2: Generating pairwise preferences...")
    pairwise_preferences = generate_pairwise_preferences(preferences, metric)

    # Save results to file
    with open("preferences.jsonl", "w") as f:
        for entry in pairwise_preferences:
            f.write(json.dumps(entry) + "\n")

    print("Pairwise preferences generated and saved to 'preferences.jsonl'.")