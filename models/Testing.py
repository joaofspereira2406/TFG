import json
from datetime import datetime
from models import Model
from Batching import batching,build_messages_batch
import torch




def test(data,model):
    name = f"{data}.json"
    batches= batching(name, batch_size=16)
    all_preds = []
    for i in range(len(batches)):
        messages = build_messages_batch(batches[i])
        predictions_list = model.classify_batch(messages)
        print(f"Processed batch {i + 1}/{len(batches)}")
        all_preds.append(predictions_list)

    return all_preds

def test_multiple_models(model_dict,datasets):
    models=model_dict.keys()
    for model_name in models:
        model=Model(model_name,model_dict[model_name])
        model.init_model()
        for data in datasets:
            predictions_list = test(data,model)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            out_name_time = now.replace(':', '-').replace(' ', '_')
            model_name=model_name.split('/')[1]
            output_path = f"../results/{model_name}_{out_name_time}.json"
            with open(output_path, 'w') as f:
                json.dump(predictions_list, f, indent=2)
                print(f"Test completed at {now}")
                print(f"Predictions saved to {output_path}")
        del model
        torch.cuda.empty_cache()
        print("Model and tokenizer removed from memory, CUDA cache cleared.")



model_dict = {
    "Qwen/Qwen2.5-3B-Instruct":"aa8e72537993ba99e69dfaafa59ed015b17504d1"
}

datasets=['BBQ-AGE']

test_multiple_models(model_dict, datasets)

