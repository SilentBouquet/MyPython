from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2Model

generator = pipeline('text-generation', model='gpt2')
set_seed(123)
generated_text = generator('hello, this is', max_length=50, num_return_sequences=3)
print(generated_text)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = 'This is the first sentence.'
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)

model = GPT2Model.from_pretrained('gpt2')
output = model(**encoded_input)
print(output['last_hidden_state'].shape)