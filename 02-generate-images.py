import os
import replicate
from concurrent.futures import ThreadPoolExecutor, as_completed

# Now that your have your trained LoRA model, you can use Flux to generate images with it. 
# Running the trained model on an H100 took on average 7 seconds ($0.0108 per image).

# To run this script, you need two things: 
# - the model trigger word you set when you trained your model
# - your model id, which is also an output from training your model, e.g. [your github username]/[your trigger word]:longrandomstringhere

# In this repo's example / demo video, I ran years = list(range(1860, 2156)) and wanted more so it was like 500 images (I added in January and July for each year)
# so that's basically $5. 
# 
# I'm going to set this to just 2021 to 2023 (6 images, $0.06) since it already cost $1 to train the model. 

replicate_api_token = input("Enter your Replicate API token (https://replicate.com/account/api-tokens): ").strip()
trigger_word = input("Enter your trigger word: ").strip()
model_id = input("Enter your model_id: ").strip()

os.environ['REPLICATE_API_TOKEN'] = replicate_api_token

output_dir = "output-v1"
os.makedirs(output_dir, exist_ok=True)

# years = list(range(1860, 2156)) 
years = list(range(2021, 2023)) 

months = ["January", "July"]  

def calculate_age(year):
    if year <= 2020:
        return round(0.5 + 29.5 * ((year - 1860) / (2020 - 1860))**1.5, 1)
    else:
        return round(30 + (40 * ((year - 2020) / (2155 - 2020))**1.2), 1)

def generate_image(year, half_year):
    age = calculate_age(year)
    if age < 1:
        age_text = f"{int(age * 12)} months old"
    else:
        age_text = f"{age} year old"
    prompt = f"A closeup headshot photo of {trigger_word}, a {age_text} male in a {half_year} {year}'s diner." # i know this prompt looks wrong but I tried making it better and this was what worked best.
    try:
        output = replicate.run(
            model_id,
            input={
                "model": "dev",
                "prompt": prompt,
                "go_fast": False,
                "lora_scale": 1,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "16:9",
                "output_format": "jpg",
                "guidance_scale": 3,
                "output_quality": 80,
                "prompt_strength": 0.8,
                "extra_lora_scale": 1,
                "num_inference_steps": 28
            }
        )
        # We want the filename outputs to sort in order like this by their name for the next part where we align faces into the video
        output_path = os.path.join(output_dir, f"{year}_{half_year.lower()}.jpg")
        with open(output_path, "wb") as f:
            f.write(output[0].read())
        print(f"Saved output for {half_year} {year} to {output_path}")
        return f"{half_year} {year}", True
    except Exception as e:
        print(f"Failed to generate image for {half_year} {year}: {e}")
        return f"{half_year} {year}", False

# I put a retry in here but none of them failed. 
def main():
    max_retries = 1
    with ThreadPoolExecutor(max_workers=10) as executor:
        tasks = {
            executor.submit(generate_image, year, month): (year, month)
            for year in years
            for month in months
        }
        results = {}

        for future in as_completed(tasks):
            half_year, success = future.result()
            results[half_year] = success

        for retry in range(max_retries):
            failed_tasks = [
                task for task, success in results.items() if not success
            ]
            if not failed_tasks:
                break
            print(f"Retrying failed tasks: {failed_tasks}")
            tasks = {
                executor.submit(generate_image, year, month): (year, month)
                for year, month in failed_tasks
            }
            for future in as_completed(tasks):
                half_year, success = future.result()
                results[half_year] = success

        failed_tasks = [task for task, success in results.items() if not success]
        if failed_tasks:
            print(f"Final failed tasks after retries: {failed_tasks}")
        else:
            print("All tasks completed successfully!")

if __name__ == "__main__":
    main()