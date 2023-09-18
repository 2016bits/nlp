import torch
import torch.nn.functional as F

class T5_Question_Answering:
    def __init__(self, model, tokenizer, gpu):
        self.model = model
        self.tokenizer = tokenizer
        self.gpu = gpu
    
    def generate(self, input_string, **generator_args):
        device = "cuda:{}".format(self.gpu) if self.gpu >= 0 else "cpu"
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(device)
        with torch.no_grad():
            res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)
    # def generate(self, input_string, **generator_args):
    #     device = "cuda:{}".format(self.gpu) if self.gpu >= 0 else "cpu"
    #     input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(device)
    #     with torch.no_grad():
    #         res = self.model.generate(input_ids, **generator_args)
    #     probs = F.softmax(res.float(), dim=1).tolist()
    #     return self.tokenizer.batch_decode(res, skip_special_tokens=True)[0].strip(), probs[0]
    
    def get_answer_from_rationale(self, rationale):
        # method 1: string matching
        indicators = ['final answer:', 'the answer is', 'the final answer is']
        split_text = ""
        for indicator in indicators:
            if rationale.find(indicator) >= 0:
                split_text = indicator

        if split_text == "":
            return None
        
        answer = rationale.split(split_text)[-1].strip()
        answer = answer[:-1] if answer.endswith('.') else answer
        return answer
    
    def answer_question(self, question):
        # moethod 2: COT
        predict_answer = {}
        example = f'Answer the following question by reasoning step-by-step.\nQ: {question}'
        out = self.generate(example, 
                    max_length = None, 
                    max_new_tokens = 128)[0].strip()
        predict_answer['rationale'] = out
        
        predict_answer['answer_text'] = self.get_answer_from_rationale(predict_answer['rationale'])

        if predict_answer['answer_text'] is None:
            example = '{}\nQ: {}\n The answer is:'.format(predict_answer['rationale'], question)
            out = self.generate(example, 
                    max_length = None, 
                    max_new_tokens = 32)[0].strip()
            predict_answer['answer_text'] = out

        return predict_answer
    
    def answer_question_directly(self, question, evidence, claim_only = False):
        if claim_only == True:
            example = f"Question: {question}\nThe answer is:"
        else:
            example = f"{evidence}\nQuestion: {question}\nThe answer is:"

        # answer question with FLAN-T5
        predict_answer = {}
        answer_text = self.generate(example, 
                    max_length = None, 
                    max_new_tokens = 32)[0].strip()
        
        predict_answer['rationale'] = ""
        predict_answer['answer_text'] = answer_text
        return predict_answer

    # def answer_verify_question(self, claim, evidence):
    #     claim = claim[:-1] if claim.endswith('.') else claim
    #     evidence = " ".join(evidence)

    #     verifiable_example = None
    #     check_example = None
    #     if evidence == []:
    #         return "Not Enough Info"
    #     else:
    #         verifiable_example = f"{evidence}\nBased on the above information, is it verifiable that {claim}? Verifiable or not verifiable? The answer is: "
    #         check_example = f"{evidence}\nBased on the above information, is it true that {claim}? True or false? The answer is: "

    #     # answer question with FLAN-T5
    #     # judge whether the claim is verifiable
    #     answer_verifiable = self.generate(verifiable_example, 
    #                 max_length = None, 
    #                 max_new_tokens = 8)[0].strip()
        
    #     verifiable_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'verifiable': 1, 'not verifiable': 0}
    #     if answer_verifiable.lower().strip() in verifiable_map:
    #         if verifiable_map[answer_verifiable.lower().strip()] == 0:
    #             return "NOT ENOUGH INFO"

    #     answer_check = self.generate(check_example,
    #                 max_length = None,
    #                 max_new_tokens = 8)[0].strip()
    #     return answer_check

    def answer_verify_question(self, claim, evidence, claim_only = False):
        claim = claim[:-1] if claim.endswith('.') else claim
        evidence = " ".join(evidence)
        if claim_only == True or evidence == []:
            return "Not Enough Info"
        else:
            example = f"{evidence}\nBased on the above information, is it true that {claim}? true, false or unknown? The answer is: "

        # answer question with FLAN-T5
        answer_text = self.generate(example, 
                    max_length = None, 
                    max_new_tokens = 8)[0].strip()
        
        return answer_text

    # def answer_verify_question(self, claim, evidence, claim_only = False):
    #     claim = claim[:-1] if claim.endswith('.') else claim
    #     evidence = " ".join(evidence)
    #     example = None
    #     if claim_only == True or evidence == []:
    #         example = f"Is it true that {claim}? True or false? The answer is: "
    #     else:
    #         example = f"{evidence}\nBased on the above information, is it true that {claim}? True or false? The answer is: "

    #     # answer question with FLAN-T5
    #     answer_text = self.generate(example, 
    #                 max_length = None, 
    #                 max_new_tokens = 8)[0].strip()
        
    #     return answer_text

