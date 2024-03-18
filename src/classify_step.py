import numpy as np
import torch
from torch import nn
import pickle as pkl
import argparse
import dataconverter


class Model(nn.Module):
    def __init__(self, num_features, num_classes, hidden_sizes):
        super(Model, self).__init__()

        # define network
        self.manual_features = nn.Sequential(
            nn.Linear(num_features, hidden_sizes[0]),
            nn.LeakyReLU(),

            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LeakyReLU(),

            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
        )

        self.clf = nn.Sequential(
            nn.Linear(hidden_sizes[2], 64),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.clf(self.manual_features(x))


def main(args):

    latest_path = args.input_file # '/home/ubuntu/new_drive/BP/saved_features/features_2023_08_23.pkl'
    X, collected_urls  = pkl.load(open(latest_path,'rb'))
    Y = [0 for _ in range(len(X))]

    model = Model(526, 2, [512, 256, 128]).to('cpu')
    model.load_state_dict(torch.load('./assets/model_at_50.pt'))
    print(model)

    print(np.array(X).shape)
    dataset = dataconverter.Dataset(
        np.array(X).squeeze(1), 
        Y, 
        {}, # url_categories 
        [], # exclusions
        collected_urls, 
        'cpu', 
        whitelist_path='./assets/whitelist.txt',
        manual_labels_files=[], 
        debug=False, 
    )

    val_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    #val_loader = dataconverter.Dataset(list(zip(np.array(X), Y, collected_urls)), batch_size=32, shuffle=False)

    print('num batches =', len(val_loader))

    result_path = latest_path.replace('.pkl', '.csv').replace('saved_', 'classify_').replace('features_', 'classify_').replace('_features', '_result')
    print(result_path)
    pred_url_labels = {}
    scam_count = 0

    whitelist = []
    with open('./assets/whitelist.txt', 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            whitelist.append(line.strip())
        fin.close()

    print('Classification Start ...')
    model.eval()
    with torch.no_grad():
        with open(result_path, 'w', encoding='utf-8') as fout:
            fout.write('URL,Label,LP,SP\n')
            for batch in val_loader:
                model_out = model(batch[0].float())

                for probs, pred, label, url in zip(torch.functional.F.softmax(model_out), model_out.argmax(dim=-1), batch[1], batch[2]):
                    
                    # unify urls
                    url = url.replace('https://', '').replace('http://', '')

                    if '/' in url:
                        url = url.split('/')[0]

                    # check the normalized url in whitelist
                    is_white = False
                    for wurl in whitelist:
                        if wurl.lower() in url.lower():
                            is_white = True
                            break

                    # write the results to file
                    fout.write('%s,%s,%0.4f,%0.4f\n' %(
                        url, 
                        'legit' if probs[1] < 0.8 or is_white else 'scam', 
                        probs[0],
                        probs[1],
                    ))
                    if pred == 1:
                        scam_count += 1
                    pred_url_labels[url] = pred.item()

                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--input_file', type=str, help='input pickle files of features', required=True)

    args = parser.parse_args()
    main(args)
