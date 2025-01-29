import os
import replicate

# I wouldn't use this script for training but you can if you prefer to use the API. 
# Instead, ignore this script and do the training at https://replicate.com/ostris/flux-dev-lora-trainer/train

def main():
    print(
        "\nIf you're using the web UI for training (recommended), skip this script and go to:"
        "\nhttps://replicate.com/ostris/flux-dev-lora-trainer/train\n"
        "- This runs on an H100 GPU ($0.001525/sec)."
        "- Training on the 20 images in this repo with the parameters below (the default) took 11m 22s ($1.04).\n"
    )

    replicate_api_token = input("Enter your Replicate API token (https://replicate.com/account/api-tokens): ").strip()
    os.environ['REPLICATE_API_TOKEN'] = replicate_api_token

    github_username = input("GitHub username: ").strip()
    model_id = input(
        "Model ID for the destination (choose a string that isn’t a real word, e.g., XDYZU293K): "
    ).strip()
    input_images_url = input("Direct download link to a .zip folder of images (e.g., from s3.amazonaws.com or a file hosting url): ").strip()

    if not input_images_url.endswith(".zip"):
        print("Input url must point to a .zip file direct download")
        return

    try:
        training = replicate.trainings.create(
            destination=f"{github_username}/{model_id.lower()}",
            version="ostris/flux-dev-lora-trainer:b6af14222e6bd9be257cbc1ea4afda3cd0503e1133083b9d1de0364d8568e6ef",
            input={
                "steps": 1000,
                "lora_rank": 16,
                "optimizer": "adamw8bit",
                "batch_size": 1,
                "resolution": "512,768,1024",
                "autocaption": True,
                "input_images": input_images_url,
                "trigger_word": model_id.upper(),  
                "learning_rate": 0.0004,
                "wandb_project": "flux_train_replicate",
                "wandb_save_interval": 100,
                "caption_dropout_rate": 0.05,
                "cache_latents_to_disk": False,
                "wandb_sample_interval": 100,
                "gradient_checkpointing": False
            },
        )

        training_url = training['urls']['get']
        model_download_url = training['output']['weights']
        model_version = training['output']['version']

        print("\nTraining initiated")
        print(f"Track progress or cancel training here: {training_url}")
        print(f"Model download url: {model_download_url}")
        print(f"Model version (copy this): {model_version}")
        print(f"Trigger word / person (also copy this): {model_id.upper()}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
