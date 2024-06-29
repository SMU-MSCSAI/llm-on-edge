from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.quantization
import os
import torch
import torch.profiler
import time
import matplotlib.pyplot as plt

def measure_inference_time(model, input_ids, attention_mask, tokenizer, model_type="unquantized"):
    log_dir = f'./log/{model_type}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        start_time = time.time()
        response_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_length=100,
            do_sample=True,      # Enable sampling to allow diverse outputs
            top_k=50,            # Consider the top 50 tokens at each step
            top_p=0.95,          # Use nucleus sampling
            temperature=0.9,     # Control the randomness of predictions
            num_return_sequences=1,  # Generate one response
            repetition_penalty=2.0   # Penalize repetitions
        )
        inference_time = time.time() - start_time
        prof.step()
    
    return inference_time, response_ids

def load_and_save_model(model_name, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        # Load the model and tokenizer from the Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Save the unquantized model and tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Model and tokenizer saved to {model_dir}")
        return model, tokenizer
    else:
        print(f"Model and tokenizer already exist in {model_dir}")
        # Load the unquantized model and tokenizer from the local directory
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        return model, tokenizer

def load_and_quantize_gpt2model(model_name, model_dir, quantized_model_dir):
    # Set the quantization backend for Apple Silicon
    torch.backends.quantized.engine = 'qnnpack'
    
    if not os.path.exists(quantized_model_dir):
        os.makedirs(quantized_model_dir)
        # Load the unquantized model and tokenizer from the local directory
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)

        # Quantize the model
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

        # Save the quantized model state_dict and tokenizer
        torch.save(quantized_model.state_dict(), f"{quantized_model_dir}/pytorch_model.bin")
        tokenizer.save_pretrained(quantized_model_dir)
        print(f"Quantized model and tokenizer saved to {quantized_model_dir}")
    else:
        print(f"Quantized model and tokenizer already exist in {quantized_model_dir}")

    # Load the quantized model and tokenizer from the local directory
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)
    quantized_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load state dict while ignoring unexpected keys
    state_dict = torch.load(f"{quantized_model_dir}/pytorch_model.bin")
    quantized_model.load_state_dict(state_dict, strict=False)
    
    return quantized_model, tokenizer

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

