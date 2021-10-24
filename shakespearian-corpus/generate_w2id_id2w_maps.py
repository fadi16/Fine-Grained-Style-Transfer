import pickle

# TODO: CMU ppl generate the vocabulary from the test data (only)

def preprocess(text_rows):
    return [row.strip().lower().split(' ') for row in text_rows]


if __name__ == "__main__":
    UNKNOWN = "<UNK>"
    PAD = "<PAD>"
    GO = "<GO>"
    EOS = "<EOS>"

    # note that in mila paper they don't include <GO> in the encoded sentances

    word_to_id = {}
    id_to_word = {}
    counter = 0

    # for special tokens
    word_to_id[UNKNOWN] = counter
    id_to_word[counter] = UNKNOWN
    counter += 1

    word_to_id[PAD] = counter
    id_to_word[counter] = PAD
    counter += 1

    word_to_id[GO] = counter
    id_to_word[counter] = GO
    counter += 1

    word_to_id[EOS] = counter
    id_to_word[counter] = EOS
    counter += 1

    input_path = "data/train.original.nltktok"
    output_path = "data/train.modern.nltktok"

    other_counter = 4

    raw_input_data = open(input_path, "r").readlines()
    print(len(raw_input_data))
    raw_output_path = open(output_path, "r").readlines()
    print(len(raw_output_path))
    preprocessed_input_data = preprocess(raw_input_data)
    preprocessed_output_data = preprocess(raw_output_path)

    for sentance_tokens in preprocessed_input_data:
        for token in sentance_tokens:
            if token not in word_to_id:
                word_to_id[token] = counter
                id_to_word[counter] = token
                counter += 1
            other_counter += 1
    for sentance_tokens in preprocessed_output_data:
        for token in sentance_tokens:
            if token not in word_to_id:
                word_to_id[token] = counter
                id_to_word[counter] = token
                counter += 1
            other_counter += 1
    print(len(word_to_id))
    print(counter)
    w2id_id2w_file = "w2id_id2w.pkl"
    outfile = open(w2id_id2w_file, "wb")
    pickle.dump((word_to_id, id_to_word), outfile)
    print(other_counter)