from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.preprocessing import normalize
from IPython.core.display import display,HTML


#Thanks to this answer: https://stackoverflow.com/questions/28907480/convert-0-1-floating-point-value-to-hex-color#28907772
def blend(color, alpha, base=[255,255,255]):
    out = [int(round((alpha * color[i]) + ((1 - alpha) * base[i]))) for i in range(3)]
    hxa = '#' + ''.join(["%02x" % e for e in out])
    return hxa

def cleantok(tok):
    return tok.replace(u'Ġ','_').replace('<','&lt;').replace('>','&gt;')

def stylize(term,colors,logit,probs=True):
    term = cleantok(term)
    color = blend(colors,logit)
    prob = str(logit)[:4]
    if logit == 1.:
        color = "#00ff00"
    if len(prob)<4:
        prob = prob + '0'
    tok = f'<span style="background-color:{color}">{term}</span>'
    if probs:
        tok += f'<sub>{prob}</sub>'
    return tok

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

question = "What are minimalist shoes"
context = """There was actually a project done on the definition of what a
minimalist shoe is and the result was "Footwear providing minimal
interference with the natural movement of the foot due to its high
flexibility, low heel to toe drop, weight and stack height, and the
absence of motion control and stability devices". If you are looking
for a simpler definition, this is what Wikipedia says, Minimalist shoes
are shoes intended to closely approximate barefoot running conditions.
1 They have reduced cushioning, thin soles, and are of lighter weight than
other running shoes, allowing for more sensory contact for the foot on the
ground while simultaneously providing the feet with some protection from
ground hazards and conditions (such as pebbles and dirt). One example of
minimalistic shoes would be the Vibram FiveFingers shoes which look like
this."""

inputs = tokenizer(question, context, add_special_tokens=True,
                   return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]
outputs = model(**inputs)
start_logits_norm = normalize(outputs[0].detach().numpy())
end_logits_norm = normalize(outputs[1].detach().numpy())

print(f'Total number of tokens: {len(input_ids)}')
print(f'Total number of start probabilities:{ start_logits_norm.shape[1]}')
print(f'Total number of end probabilities:{ end_logits_norm.shape[1]}')


start_toks = []
end_toks = []
terms = tokenizer.convert_ids_to_tokens(input_ids)
start_token_id = 0
end_token_id = len(terms)
for i in range(len(terms)):
    print(terms[i])
    start_toks.append(stylize(terms[i],[0,127,255],start_logits_norm[0][i]))
    end_toks.append(stylize(terms[i],[255,0,255],end_logits_norm[0][i]))
    if start_logits_norm[0][i]==1.:
        start_token_id = i
    if end_logits_norm[0][i]==1.:
        end_token_id = i+1
answer = terms[start_token_id:end_token_id]

print(cleantok(' '.join(answer)))
print(' '.join(start_toks))
print(' '.join(end_toks))
