
import logging
from helpers import load_and_save_model, quantize_model_dynamic, quantize_model_static, generate_response, measure_inference_time, get_model_size_mb


def load_model(model_name, model_dir):
    logging.info("Loading the model...")
    try:
        model, tokenizer = load_and_save_model(model_name, model_dir)
        logging.info("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load the model: {e}")
        exit()

def quantize_model(model, tokenizer, quantized_model_dir, quantization_type="dynamic"):
    logging.info("Quantizing the model...")
    if quantization_type == "dynamic":
        try:
            # quantized_model = quantize_model_dynamic2(model, unq_model_dir, quantized_model_dir)
            quantized_model = quantize_model_dynamic(model, tokenizer, quantized_model_dir)
            logging.info("Model quantized successfully.")
            return quantized_model
        except Exception as e:
            logging.error(f"Failed to quantize the model: {e}")
            exit()
    elif quantization_type == "static":
        # Define calibration data (a few example sentences) for static quantization
        calibration_data = [
            "This is a sample sentence for calibration.",
            "Another example sentence to calibrate the model.",
            "Using various sentences helps improve calibration accuracy."
        ]
        try:
            quantized_model = quantize_model_static(model, tokenizer, quantized_model_dir, calibration_data)
            logging.info("Model quantized successfully.")
            return quantized_model
        except Exception as e:
            logging.error(f"Failed to quantize the model: {e}")
            exit()

def generate_and_print_response(quantized_model, tokenizer, prompt):
    try:
        response = generate_response(quantized_model, tokenizer, prompt)
        print(f"Response:\n{response}")
    except Exception as e:
        logging.error(f"Failed to generate response: {e}")

def measure_and_print_inference_time(quantized_model, tokenizer, prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inference_time, response_ids = measure_inference_time(quantized_model, inputs["input_ids"], inputs["attention_mask"], tokenizer, model_type="quantized")
        print(f"Inference Time: {inference_time} seconds")
    except Exception as e:
        logging.error(f"Failed to measure inference time: {e}")

def print_model_size(quantized_model_dir, unquantized_model_dir):
    try:
        q_model_size = get_model_size_mb(quantized_model_dir)
        uq_model_size = get_model_size_mb(unquantized_model_dir)
        
        print(f"Quantized model size: {q_model_size} MB")
        print(f"In GB: {q_model_size / 1024} GB")
        
        print(f"Unquantized model size: {uq_model_size} MB")
        print(f"In GB: {uq_model_size / 1024} GB") if uq_model_size else None
        
        print(f"Model size reduced by: {uq_model_size - q_model_size} MB")
        print(f"Memory footprint reduced by: {uq_model_size / q_model_size} times")
    except Exception as e:
        logging.error(f"Failed to get model size: {e}")


def main():
    # Set up logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    ## Try with TinyLlama Model ###
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # or "gpt2" for GPT-2
    model_dir = "/Users/tangotew/Documents/repos/AI/llm-on-edge/models/tiny-llama-1.1B"  # or "../models/gpt2"
    quantized_model_dir = "/Users/tangotew/Documents/repos/AI/llm-on-edge/models/tiny-llama-1.1B-quantized"  # or "../models/gpt2-quantized"
    
    ## Try with GPT2 Model ###
    # model_name = "gpt2"
    # model_dir = "/Users/tangotew/Documents/repos/AI/llm-on-edge/models/gpt2"
    # quantized_model_dir = "/Users/tangotew/Documents/repos/AI/llm-on-edge/models/gpt2-quantized"
    
    # ## Try with Gemma Model ###
    # model_name = "Google/Gemma-2-9b"
    # model_dir = "/Users/tangotew/Documents/repos/AI/llm-on-edge/models/gemma-2B"
    # quantized_model_dir = "/Users/tangotew/Documents/repos/AI/llm-on-edge/models/gemma-2B-quantized"

    # Load or download the model
    model, tokenizer = load_model(model_name, model_dir)

    # Quantize the model
    quantized_model = quantize_model(model, tokenizer, quantized_model_dir, quantization_type="dynamic")

    # Generate a response
    generate_and_print_response(quantized_model, tokenizer, "Hello, How are you!.")

    # Measure inference time
    # measure_and_print_inference_time(quantized_model, tokenizer, "Write a Python function to fetch weather data from an API.")

    print_model_size(quantized_model_dir, model_dir)
    
    
if __name__ == "__main__":
    main()
