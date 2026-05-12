# Each line is "<weight> <prefix>". The prefix points to an indexed-dataset
# pair "<prefix>.bin" + "<prefix>.idx" produced by tools/data/preprocess_data.py.
# Weights are normalised automatically; only the ratio matters.
export DATA_PATH="0.5 /path/to/data1_text_document
0.5 /path/to/data2_text_document"
