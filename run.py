
# Authenticate with Hugging Face
TOKEN = "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"
print("Logging into Hugging Face...")
login(TOKEN)

# Clone the Parler-TTS repository if it doesn't exist
repo_dir = "parler-tts"
if not os.path.exists(repo_dir):
    print("Cloning Parler-TTS repository...")
    subprocess.run(["git", "clone", "https://github.com/huggingface/parler-tts.git"], check=True)

# Change into the repo directory
os.chdir(repo_dir)

# Install dependencies
print("Installing dependencies...")
subprocess.run(["pip", "install", "-e", ".[train]"], check=True)

# Configure Accelerate
print("Configuring Accelerate...")
subprocess.run(["accelerate", "config"], check=True)

# Run the training script
print("Starting training...")
subprocess.run([
    "accelerate", "launch", "./training/run_parler_tts_training.py",
    "--model_name_or_path", "parler-tts/parler_tts_mini_v0.1",
    "--feature_extractor_name", "parler-tts/dac_44khZ_8kbps",
    "--description_tokenizer_name", "parler-tts/parler-tts-mini-v1",
    "--prompt_tokenizer_name", "parler-tts/parler-tts-mini-v1",
    "--report_to", "wandb",
    "--overwrite_output_dir", "true",
    "--train_dataset_name", "Hidi-agili/yoruba_dataset",
    "--train_metadata_dataset_name", "Kejihela/yoruba-TTS",
    "--train_dataset_config_name", "default",
    "--train_split_name", "train",
    "--eval_dataset_name", "Hidi-agili/yoruba_dataset",
    "--eval_metadata_dataset_name", "Kejihela/yoruba-TTS",
    "--eval_dataset_config_name", "default",
    "--eval_split_name", "train",
    "--max_eval_samples", "8",
    "--per_device_eval_batch_size", "8",
    "--target_audio_column_name", "audio",
    "--description_column_name", "pesq_speech_quality",
    "--prompt_column_name", "text",
    "--max_duration_in_seconds", "20",
    "--min_duration_in_seconds", "2.0",
    "--max_text_length", "400",
    "--preprocessing_num_workers", "2",
    "--do_train", "true",
    "--num_train_epochs", "2",
    "--gradient_accumulation_steps", "18",
    "--gradient_checkpointing", "true",
    "--per_device_train_batch_size", "2",
    "--learning_rate", "0.0001",
    "--adam_beta1", "0.9",
    "--adam_beta2", "0.99",
    "--weight_decay", "0.01",
    "--lr_scheduler_type", "constant_with_warmup",
    "--warmup_steps", "50",
    "--logging_steps", "2",
    "--freeze_text_encoder", "true",
    "--audio_encoder_per_device_batch_size", "5",
    "--dtype", "float16",
    "--seed", "456",
    "--output_dir", "./output_dir_training/",
    "--temporary_save_to_disk", "./audio_code_tmp/",
    "--save_to_disk", "./tmp_dataset_audio/",
    "--dataloader_num_workers", "2",
    "--do_eval",
    "--predict_with_generate",
    "--include_inputs_for_metrics",
    "--group_by_length", "true"
], check=True)

print("Training completed successfully!")
