import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from retrieval.const import BASE_MODEL_NAME
from retrieval.dataset import pad_tensors
from retrieval.model import RetrievalModel

if __name__ == '__main__':
    model = RetrievalModel().eval()
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        # model.load_state_dict(load_file('models/retrieval-200.safetensors'))

        query = 'Which American-born Sinclair won the Nobel Prize for Literature in 1930?'
        query = tokenizer.encode_plus(query).encodings[0]
        documents = [
            'Sinclair Lewis Sinclair Lewis Harry Sinclair Lewis (February 7, 1885 – January 10, 1951) was an American novelist, short-story writer, and playwright. In 1930, he became the first writer from the United States to receive the Nobel Prize in Literature, which was awarded "for his vigorous and graphic art of description and his ability to create, with wit and humor, new types of characters." His works are known for their insightful and critical views of American capitalism and materialism between the wars. He is also respected for his strong characterizations of modern working women. H. L. Mencken wrote of him, "[If] there',
            'Judi Dench regular contact with the theatre. Her father, a physician, was also the GP for the York theatre, and her mother was its wardrobe mistress. Actors often stayed in the Dench household. During these years, Judi Dench was involved on a non-professional basis in the first three productions of the modern revival of the York Mystery Plays in 1951, 1954 and 1957. In the third production she played the role of the Virgin Mary, performed on a fixed stage in the Museum Gardens. Though she initially trained as a set designer, she became interested in drama school as her brother Jeff',
            'Corruption in Angola they really are. Angola\'s colonial era ended with the Angolan War of Independence against Portugal occurred between 1970 and 1975. Independence did not produce a unified Angola, however; the country plunged into years of civil war between the National Union for the Total Independence of Angola (UNITA) and the governing Popular Movement for the Liberation of Angola (MPLA). 30 years of war would produce historical legacies that combine to allow for the persistence of a highly corrupt government system. The Angolan civil war was fought between the pro-western UNITA and the communist MPLA and had the characteristics typical of a'
        ]
        documents = tokenizer(documents).encodings
        document_input_ids = pad_tensors([torch.tensor(x.ids) for x in documents])
        document_attention_mask = pad_tensors([torch.tensor(x.attention_mask) for x in documents])

        query_vec = model(torch.tensor(query.ids)[None, :], torch.tensor(query.attention_mask)[None, :], is_query=True)
        document_vec = model(document_input_ids, document_attention_mask, is_query=False)

        cosine_similarities = query_vec @ document_vec.T


        print(123)