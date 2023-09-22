import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor


class FlagAIModel:
    def __init__(self, args):
        loader = AutoLoader(args.task_name,
                            model_name=args.model_name,
                            model_dir=args.model_path,
                            use_cache=args.use_cache,
                            fp16=args.fp16)
        model = loader.get_model()
        model.half()
        model.eval()
        if args.use_cuda:
            model.cuda()

        self.max_new_tokens = args.max_new_tokens
        self.tokenizer = loader.get_tokenizer()
        self.predictor = Predictor(model, self.tokenizer)
    
    def generate_program(self, prompt, temperature=1.0, top_p=1.0):
        input_ids = self.tokenizer.encode_plus_non_glm(prompt)["input_ids"][:-1]
        input_len = len(input_ids)

        max_length = input_len + self.max_new_tokens
        with torch.no_grad():
            res = self.predictor.predict_generate_randomsample(
                prompt, out_max_length=max_length, top_p=top_p, temperature=temperature
            )
        # remove excess generated parts from the tail
        res = res.split('\n\n')[0]
        return res
    
    def generate_answer(self, prompt, temperature=0.0, top_p=1.0):
        input_ids = self.tokenizer.encode_plus_non_glm(prompt)["input_ids"][:-1]
        input_len = len(input_ids)

        max_length = input_len + self.max_new_tokens
        with torch.no_grad():
            res = self.predictor.predict_generate_randomsample(
                prompt, out_max_length=max_length, top_p=top_p, temperature=temperature
            )
        # # remove excess generated parts from the tail
        # res = res.split()
        # return res[0] if res[0] else res[1]
        return res
