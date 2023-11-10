from allennlp.predictors import Predictor

predictor = Predictor.from_path("/data/yangjun/tools/elmo-constituency-parser-2018.03.14.tar.gz")
# predictor = Predictor.from_path(
#     "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")

claim = "The New Jersey Turnpike has zero shoulders."
tokens = predictor.predict(claim)
print(tokens)
