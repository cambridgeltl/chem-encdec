from transformers import GenerationConfig
import argparse
import sys

def set_generation_config(args):

    # Set generation config
    if args.infer_mode == "greedy":
        generation_config = GenerationConfig.from_pretrained(
            args.model_path,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            renormalize_logits=True,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
        print("Using greedy!")

    elif args.infer_mode == "search_beam":
         generation_config = GenerationConfig.from_pretrained(
             args.model_path,
             num_beams=args.num_beams,
             ido_sample=False,
             return_dict_in_generate=True,
             remove_invalid_values=True,
             output_scores=True,
             renormalize_logits=True,
             max_new_tokens=args.max_new_tokens,
             num_return_sequences=args.num_return_sequences,
             use_cache=True,
         )
         print("Using basic beam search!")

    elif args.infer_mode == "nucleus":
         generation_config = GenerationConfig.from_pretrained(
             args.model_path,
             top_k=args.top_k,
             top_p=args.top_p,
             do_sample=True,
             remove_invalid_values=True,
             renormalize_logits=True,
             return_dict_in_generate=True,
             output_scores=True,
             max_new_tokens=args.max_new_tokens,
             num_return_sequences=args.num_return_sequences,
             use_cache=True
         )
         print("Using nucleus sampling!")

    elif args.infer_mode == "diverse_beam":
        generation_config = GenerationConfig.from_pretrained(
            args.model_path,
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
            diversity_penalty=args.diversity_penalty,
            remove_invalid_values=True,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True
        )
        print("Using diverse beam search")

    elif args.infer_mode == "sampling_beam":
        generation_config = GenerationConfig.from_pretrained(
            args.model_path,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=True,
            length_penalty=args.length_penalty,
            remove_invalid_values=True,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True
        )
        print("Using beam sampling")

    elif args.infer_mode == "contrastive":
        generation_config = GenerationConfig.from_pretrained(
            args.model_path,
            penalty_alpha=args.penalty_alpha,
            top_k = args.top_k,
            remove_invalid_values=True,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True
        )
        print("Using contrastive search")

    else:
        print("(Currently) unsupported inference mode!")
        sys.exit(-1)

    return generation_config

