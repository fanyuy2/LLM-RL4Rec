import json
from transformers import pipeline
from itertools import combinations

def sample_and_evaluate(model_id, prompts, k=5, max_new_tokens=256):
    # Load model and tokenizer
    test_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=test_model, tokenizer=tokenizer, device="cuda")

    # Store sampled outputs and metrics
    preference_data = []

    for prompt in prompts:
        # Sample k outputs for the prompt
        samples = []
        for _ in range(k):
            output = pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
            generated_text = output[0]['generated_text']
            samples.append(generated_text)

        # Evaluate metrics for each sampled output
        eval_results = []
        for sample in samples:
            llm_generated_movies = extract_movie_titles(sample)  # Your movie extraction function
            matched_titles_with_scores = match_titles_batch(llm_generated_movies, all_movies)
            recommended_movies = [title for title, score in matched_titles_with_scores if title]
            metrics = evaluate_recommendations(ground_truth=[], recommended_movies=recommended_movies, k=k)  # Replace ground_truth
            eval_results.append(metrics)

        # Store the data
        preference_data.append({
            "prompt": prompt,
            "samples": [{"text": sample, "metrics": metrics} for sample, metrics in zip(samples, eval_results)]
        })

    return preference_data


def generate_preference_data(preference_data, metric_key):
    """
    Generate preference data from sampled outputs and their associated metrics.
    Args:
        preference_data: List of dictionaries with prompts, outputs, and metrics.
        metric_key: The key of the metric to use for preference comparison.
    Returns:
        List of preference data in the format:
        [{"prompt": prompt, "response_1": text_1, "response_2": text_2, "preference": int}]
    """
    preference_dataset = []

    for item in preference_data:
        prompt = item["prompt"]
        samples = item["samples"]

        # Generate pairwise comparisons (kC2)
        for (sample_1, sample_2) in combinations(samples, 2):
            response_1 = sample_1["text"]
            response_2 = sample_2["text"]
            metric_1 = sample_1["metrics"][metric_key]
            metric_2 = sample_2["metrics"][metric_key]

            # Determine preference based on the metric
            if metric_1 > metric_2:  # You can adjust this logic for specific metric meanings
                preference = 1  # response_1 > response_2
            elif metric_1 < metric_2:
                preference = -1  # response_2 > response_1
            else:
                preference = 0  # No preference

            # Append to preference dataset
            preference_dataset.append({
                "prompt": prompt,
                "response_1": response_1,
                "response_2": response_2,
                "preference": preference,
                "metrics_1": sample_1["metrics"],
                "metrics_2": sample_2["metrics"]
            })

    return preference_dataset


# Step 1: Generate samples and evaluate
prompts = ["Recommend some movies for a rainy day.", "What are some great thrillers from the 90s?"]
preference_data = sample_and_evaluate(model_id="your-model-id", prompts=prompts, k=5)

# Save preference data for analysis
with open("preference_data.json", "w") as f:
    json.dump(preference_data, f, indent=4)

# Step 2: Generate pairwise preference dataset
metric_key = "precision@5"  # Replace with your chosen metric
pairwise_preferences = generate_preference_data(preference_data, metric_key)

# Save pairwise preferences
with open("pairwise_preferences.json", "w") as f:
    json.dump(pairwise_preferences, f, indent=4)



