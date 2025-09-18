import h5py
import json
import warnings

# Suppress FutureWarnings from h5py
warnings.simplefilter(action='ignore', category=FutureWarning)

MODEL_PATH = 'disease_classifier.h5'

print(f"--- Inspecting Model File: {MODEL_PATH} ---")

try:
    with h5py.File(MODEL_PATH, 'r') as f:
        # Print top-level attributes, which contain the configuration
        print("\n[+] Top-level attributes:")
        for key, value in f.attrs.items():
            print(f"  - {key}")

        # Decode and print the model configuration JSON
        if 'model_config' in f.attrs:
            print("\n[+] Model Configuration (JSON):")
            # Fix: The attribute is already a string, so no decoding is needed.
            model_config_json = f.attrs['model_config']
            model_config = json.loads(model_config_json)
            
            # Pretty-print the JSON structure
            print(json.dumps(model_config, indent=2))
        else:
            print("\n[-] No 'model_config' attribute found in the file's metadata.")

    print("\n--- Inspection Complete ---")
    print("\nIMPORTANT: Please copy and paste the entire output, especially the 'Model Configuration' section.")

except Exception as e:
    print(f"\n❌ FAILED: An error occurred while inspecting the file.")
    print(f"Error: {e}")

