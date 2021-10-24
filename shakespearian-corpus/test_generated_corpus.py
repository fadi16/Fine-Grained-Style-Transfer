import pickle
from platform import win32_ver

if __name__ == "__main__":
    w2id, id2w = pickle.load(open("w2id_id2w.pkl", "rb"))
    print(len(w2id))
    tokens_counter = 0
    for mode in ["test"]: #["train", "test", "valid"]:
        print(">>>" + mode + "<<<")
        encoded_sentances, classes = pickle.load(open(mode + "_C.pkl", "rb"))
        for encoded_sentance in encoded_sentances:
            for token_id in encoded_sentance:
                if token_id > 17000:
                    print(token_id)