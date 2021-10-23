import pickle

UNKNOWN = "<UNK>"
PAD = "<PAD>"
GO = "<GO>"
EOS = "<EOS>"

def preprocess(text_rows):
    return [row.strip().lower().split(' ') for row in text_rows]

def encode_sentances(mode="train"):
    input_path = "data/{0}.original.nltktok".format(mode)
    output_path = "data/{0}.modern.nltktok".format(mode)

    raw_input_data = open(input_path, "r").readlines()
    raw_output_data = open(output_path, "r").readlines()

    preprocessed_input_sentances = preprocess(raw_input_data)
    preprocessed_output_sentances = preprocess(raw_output_data)

    # get word to id and id to word maps
    w2id, id2w = pickle.load(open("w2id_id2w.pkl", "rb"))

    all_sentances_codes = []
    all_sentances_classes = []

    # shakeapearian
    for preprocessed_sentance in preprocessed_input_sentances:
        # todo: mila preprocessed data does not have go symbols
        sentance_codes = []
        for token in preprocessed_sentance:
            if token in w2id:
                sentance_codes.append(w2id[token])
            else:
                sentance_codes.append(w2id[UNKNOWN])
        sentance_codes.append(w2id[EOS])
        all_sentances_codes.append(sentance_codes)
        all_sentances_classes.append([0,1])

    # modern
    for preprocessed_sentance in preprocessed_output_sentances:
        # todo: mila preprocessed data does not have go symbols
        sentance_codes = []
        for token in preprocessed_sentance:
            if token in w2id:
                sentance_codes.append(w2id[token])
            else:
                sentance_codes.append(w2id[UNKNOWN])
        sentance_codes.append(w2id[EOS])
        all_sentances_codes.append(sentance_codes)
        all_sentances_classes.append([1,0])

    # TODO: there seems to be no padding in mila preprocessed data
    output_pickle_file = "{0}.pkl".format(mode)
    outfile = open(output_pickle_file, "wb")
    pickle.dump(all_sentances_codes, outfile)

    # now dump both encoded sentances and classes
    output_pickle_file = "{0}_C.pkl".format(mode)
    outfile = open(output_pickle_file, "wb")
    pickle.dump((all_sentances_codes, all_sentances_classes), outfile)

if __name__ == "__main__":
    encode_sentances("train")
    encode_sentances("valid")
    encode_sentances("test")
