import os
import re
from huggingface_hub import InferenceClient

# Retrieve the API key from an environment variable
api_key = os.getenv("HUGGINGFACE_API_KEY")
if api_key is None:
    raise ValueError("API key for Hugging Face is not set. Please set 'HUGGINGFACE_API_KEY' in your environment.")

# Initialize the API client with the retrieved API key
client = InferenceClient(api_key=api_key)


def get_llm_recommendations(prompt, model_name="meta-llama/Llama-3.2-1B-Instruct", max_tokens=500, top_k=5):
    """
    Sends the prompt to the LLM and retrieves the recommended movie list.

    Parameters:
    - prompt (str): The input prompt generated for the LLM.
    - model_name (str): The name of the model to use for recommendations.
    - max_tokens (int): Maximum tokens to allow in the output.
    - top_k (int): Number of movie recommendations to retrieve.

    Returns:
    - list of str: A list of recommended movie titles parsed from the LLM's output.
    """
    try:
        # Send the prompt to the LLM
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=False
        )

        # Extract the response text
        llm_output = response['choices'][0]['message']['content'].strip()
        # print(llm_output)
        # Use regex to extract the list of movie titles from the output
        movie_list_match = re.search(r"\[.*?\]", llm_output, re.DOTALL)
        if movie_list_match:
            # Evaluate the extracted string to a Python list
            recommendations = eval(movie_list_match.group())
            if isinstance(recommendations, list) and all(isinstance(item, str) for item in recommendations):
                return recommendations[:top_k]
            else:
                # return []
                raise ValueError("LLM response format is incorrect")
        else:
            print("No valid list of movie titles found in the output.")
            return []

    except Exception as e:
        print(f"Error during LLM API request or response parsing: {e}")
        return []

if __name__ == "__main__":
    # Example usage with updated prompt
    prompt = """I am a male, aged 24 from Los Angeles, CA working as a Fashion Designer.
    I have previously watched and liked the movies: ['Inception (Sci-Fi, Thriller)', 'Wonder Woman (Action, Adventure)'].
    I have also watched and disliked the movies: ['The Shallows (Thriller)', 'Shoplifting (Crime, Drama)'].
    Based on my history, recommend the top 5 movies I am most likely to watch next.
    Please provide the output in a list of strings format, containing only the movie titles.
    Make sure to strictly adhere to the output format given below. Strictly do not generate any additional information other than the movie names.
    Format:  ['movie_name', 'movie_name', ... 'movie_name']
    Make sure to limit the recommendations to movies available in the MovieLens dataset."""

    # Call the function with the generated prompt
    recommended_movies = get_llm_recommendations(prompt)
    print(recommended_movies)
