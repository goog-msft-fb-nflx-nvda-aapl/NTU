import sys
sys.path.append(".")

import os
import json
import argparse
import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling

evolve_vcd_sampling()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--images_root", type=str, required=True)
    parser.add_argument("--llava_weight_path", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--use_vcd", action="store_true", default=False)
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--noise_step", type=int, default=500)
    return parser.parse_args()


def postprocess(output):
    output = output.strip()
    if output.lower().startswith("yes"):
        return "Yes"
    elif output.lower().startswith("no"):
        return "No"
    # fallback: check first word
    first = output.split()[0].lower().rstrip(".,!?")
    if first == "yes":
        return "Yes"
    return "No"


def main():
    args = parse_args()

    model_path = args.llava_weight_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    model.eval()

    with open(args.annotation_file, "r") as f:
        data = json.load(f)

    conv_mode = "llava_v1"

    results = []

    for item in data:
        image_source = item["image_source"]
        question = item["question"]
        answer = item["answer"]

        image_path = os.path.join(args.images_root, image_source + ".jpg")
        image = Image.open(image_path).convert("RGB")

        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # noisy image for VCD
        image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        image_tensor_cd = image_tensor_cd.to(model.device, dtype=torch.float16)

        # build prompt
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question + " Please answer with Yes or No."
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        gen_kwargs = dict(
            images=image_tensor,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=64,
            stopping_criteria=[stopping_criteria],
        )
        if args.use_vcd:
            gen_kwargs["images_cd"] = image_tensor_cd
            gen_kwargs["cd_alpha"] = args.cd_alpha
            gen_kwargs["cd_beta"] = args.cd_beta

        with torch.inference_mode():
            output_ids = model.generate(input_ids, **gen_kwargs)

        output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        if output.endswith(stop_str):
            output = output[: -len(stop_str)].strip()

        pred = postprocess(output)

        results.append({
            "image_source": image_source,
            "question": question,
            "answer": answer,
            "prediction": pred,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.pred_file)), exist_ok=True)
    with open(args.pred_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} predictions to {args.pred_file}")

    # quick accuracy
    correct = sum(1 for r in results if r["prediction"].lower() == r["answer"].lower())
    print(f"Accuracy: {correct}/{len(results)} = {correct/len(results):.4f}")


if __name__ == "__main__":
    main()