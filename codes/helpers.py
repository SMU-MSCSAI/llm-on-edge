from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.quantization
import os
import torch
import torch.profiler
import time
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from torch.quantization import float_qparams_weight_only_qconfig, prepare, convert
from huggingface_hub import login
from dotenv import load_dotenv

# Function to measure inference time and profile the model
def measure_inference_time(model, input_ids, attention_mask, tokenizer, model_type="unquantized"):
    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

    log_dir = f'./log/{model_type}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=schedule(wait=1, warmup=1, active=3),
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        start_time = time.perf_counter()
        response_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_length=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=1,
            repetition_penalty=2.0
        )
        inference_time = time.perf_counter() - start_time
        prof.step()
    
    return inference_time, response_ids

# Function to load and save the model
def load_and_save_model(model_name, model_dir):
    if not os.path.exists(model_dir) or not os.path.exists(f"{model_dir}/config.json"):
        os.makedirs(model_dir, exist_ok=True)
        try:
            # Display progress bar for loading the model and tokenizer
            with tqdm(total=2, desc="Loading model and tokenizer", unit="step") as pbar:
                # Load the model and tokenizer from the Hugging Face Hub
                if model_name.lower().startswith("gemma"):
                    hf_token, hf_username = login_to_huggingface()
                    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                    pbar.update(1)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    pbar.update(1)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                pbar.update(1)
        except Exception as e:
            logging.error(f"Failed to load the model: {e}")
        # Save the unquantized model and tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        logging.info(f"Model and tokenizer saved to {model_dir}")
        return model, tokenizer
    else:
        print(f"Model and tokenizer already exist in {model_dir}")
        try:
            with tqdm(total=2, desc="Loading model and tokenizer", unit="step") as pbar:
                # Load the unquantized model and tokenizer from the local directory for gemma models
                if model_name.lower().startswith("gemma"):
                    hf_token, hf_username = login_to_huggingface()
                    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                    pbar.write("Loaded tokenizer")
                    pbar.update(1)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    pbar.write("Loaded model")
                    pbar.update(1)
                # load general models from the local directory(gp2, gpt-neo, etc)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    pbar.update(1)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    pbar.update(1)
                
            return model, tokenizer
        except MemoryError as e:
            logging.error(f"Failed to load the model due to memory insufficiency: {e}")
            raise
        except Exception as e:
            logging.error(f"Failed to load the model: {e}")
            raise

def quantize_model_dynamic(model, tokenizer, quantized_model_dir):
    # Set the quantization backend for Apple Silicon
    torch.backends.quantized.engine = 'qnnpack'

    if not os.path.exists(quantized_model_dir):
        os.makedirs(quantized_model_dir, exist_ok=True)
        print("Setting up dynamic quantization...")
        try:
            logging.info("Quantizing the model...")

            # Apply dynamic quantization
            with tqdm(total=1, desc="Quantizing model", unit="step") as pbar:
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                pbar.update(1)

            logging.info("Model quantized successfully.")
            logging.info(f"Saving quantized model and tokenizer to {quantized_model_dir}")

            # Save the quantized model and tokenizer
            model_to_save = quantized_model.module if hasattr(quantized_model, 'module') else quantized_model
            torch.save(model_to_save.state_dict(), f"{quantized_model_dir}/pytorch_model.bin")
            model.config.save_pretrained(quantized_model_dir)
            tokenizer.save_pretrained(quantized_model_dir)
            logging.info(f"Quantized model and tokenizer saved to {quantized_model_dir}")
        except Exception as e:
            logging.error(f"Failed to quantize the model: {e}")
            raise
    # If the quantized model already exists, load it
    else:
        print(f"Quantized model already exists in {quantized_model_dir}")
        try:
            model_config = AutoModelForCausalLM.from_pretrained(quantized_model_dir).config
            quantized_model = AutoModelForCausalLM.from_config(model_config)
            quantized_model.load_state_dict(torch.load(f"{quantized_model_dir}/pytorch_model.bin"))
            return quantized_model
        except Exception as e:
            logging.error(f"Failed to load the quantized model: {e}")
            raise
    return quantized_model

# Function for static quantization
def quantize_model_static(model, tokenizer, quantized_model_dir, calibration_data):
    # Set the quantization backend for Apple Silicon
    torch.backends.quantized.engine = 'qnnpack'
    
    if not os.path.exists(quantized_model_dir):
        os.makedirs(quantized_model_dir, exist_ok=True)
    
    print("Setting up static quantization...")
    try:
        # Prepare the model for static quantization with the correct qconfig
        model.qconfig = float_qparams_weight_only_qconfig
        for name, module in model.named_modules():
            if 'embeddings' in name:
                module.qconfig = None  # Disable quantization for embeddings
        prepare(model, inplace=True)
        
        # Calibrate the model with sample data
        print("Calibrating the model...")
        model.eval()
        with torch.no_grad():
            for batch in tqdm(calibration_data, desc="Calibrating", unit="batch"):
                inputs = tokenizer(batch, return_tensors='pt')
                model(inputs['input_ids'])
        
        # Convert the model to a quantized version
        print("Converting the model to quantized version...")
        quantized_model = convert(model, inplace=True)

        # Save the quantized model and tokenizer
        quantized_model.save_pretrained(quantized_model_dir)
        tokenizer.save_pretrained(quantized_model_dir)
        print(f"Quantized model and tokenizer saved to {quantized_model_dir}")
    except Exception as e:
        print(f"An error occurred during quantization: {e}")
        raise

    return quantized_model

def plot_measurements(unquantized_model_size, quantized_model_size, unquantized_inference_time, quantized_inference_time):
    labels = ['Unquantized', 'Quantized']
    
    sizes = [unquantized_model_size, quantized_model_size]
    inference_times = [unquantized_inference_time, quantized_inference_time]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot model sizes
    ax1.bar(labels, sizes, color=['blue', 'green'])
    ax1.set_title('Model Size Comparison')
    ax1.set_ylabel('Size (MB)')

    # Plot inference times
    ax2.bar(labels, inference_times, color=['blue', 'green'])
    ax2.set_title('Inference Time Comparison')
    ax2.set_ylabel('Time (seconds)')

    plt.show()

# Function to generate a response
def generate_response(model, tokenizer, prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_length, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, num_return_sequences=1, repetition_penalty=2.0)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Function to get model size in MB
def get_model_size_mb(model_dir):
    total_size = 0
    for file in os.listdir(model_dir):
        if os.path.isfile(os.path.join(model_dir, file)):
            total_size += os.path.getsize(os.path.join(model_dir, file))
    return total_size / (1024 * 1024) 

def login_to_huggingface():
    load_dotenv()  # Load environment variables from .env file
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    
    if not hf_token or not hf_username:
        raise ValueError("Hugging Face token and username must be set in the .env file")
    
    login(token=hf_token)
    return hf_token, hf_username