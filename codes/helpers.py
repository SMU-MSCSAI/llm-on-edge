from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.quantization
import os
import torch

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

def load_and_quantize_model(model_name, model_dir, quantized_model_dir):
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
    quantized_model.load_state_dict(torch.load(f"{quantized_model_dir}/pytorch_model.bin"))
    return quantized_model, tokenizer

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