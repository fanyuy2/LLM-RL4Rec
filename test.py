import itertools
import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

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
    for data in tqdm(dataset["train"]):
        prompt = data["prompt"]
        
        # Generate k responses
        responses = []
        for _ in range(k):
            output = pipe(prompt, max_new_tokens=256, num_return_sequences=1)
            response_text = output[0]["generated_text"]
            responses.append(response_text)
        
        # Evaluate each response
        metrics = []
        for response in responses:
            # Assuming `evaluate_recommendations` is your evaluation function
            # Replace this with your actual evaluation logic
            metric_value = evaluate_recommendations(data["completion"], response, metric)
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
    for item in preferences:
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
    # Load your dataset (replace with actual dataset loading logic)
    dataset = {
        "train": [
            {"prompt": "Recommend me some action movies.", "completion": "Mad Max, John Wick, Die Hard"},
            {"prompt": "Suggest me romantic comedies.", "completion": "Crazy Rich Asians, Notting Hill, Love Actually"}
        ]
    }
    
    # Generate samples and evaluate
    model_id = "gpt-3.5-turbo"
    k = 5
    preferences = generate_and_evaluate_samples(model_id, dataset, k, metric="precision")
    
    # Generate pairwise preferences
    pairwise_preferences = generate_pairwise_preferences(preferences, metric="precision")
    
    # Save results to file
    with open("preferences.jsonl", "w") as f:
        for entry in pairwise_preferences:
            f.write(json.dumps(entry) + "\n")

    print("Pairwise preferences generated and saved!")
