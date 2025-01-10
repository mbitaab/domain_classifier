import numpy as np
import pickle as pkl
import argparse


def normalize_url(url):
    prefixes = ['https://www.', 'http://www.', 'www.']
    for prefix in prefixes:
        if url.startswith(prefix):
            return url[len(prefix):]
    return url

def is_domain_whitelisted(domain, whitelist):
    domain_parts = domain.split('.')
    if domain in whitelist:
        return True

    if len(domain_parts) > 1:
        tld = '.'.join(domain_parts[-2:])
        if tld in whitelist:
            return True
    return False

def main(args):
    latest_path = args.input_file # '/home/ubuntu/new_drive/BP/saved_features/features_2023_08_23.pkl'
    result_path = args.output_file
    
    X, collected_urls  = pkl.load(open(latest_path,'rb'))
    Y = [0 for _ in range(len(X))]

    X = np.array(X).squeeze(1)
    model = pkl.load(open('./assets/xgb_model.pkl', 'rb'))

    print(np.array(X).shape)

    pred_url_labels = {}
    scam_count = 0

    whitelist = []
    with open('/usr/src/app/assets/whitelist.txt', 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            whitelist.append(line.strip().lower())
        fin.close()

    print('Classification Start ...')
    with open(result_path, 'w', encoding='utf-8') as fout:
        fout.write('URL,Label,LP,SP\n')
        
        model_out = model.predict_proba(X)

        for probs, pred, label, url in zip( model_out,(model_out[:, 1] >= args.threshold).astype(int), Y, collected_urls ):
            print(f'DOMAIN_CLASSIFIER:{probs[0]},{probs[1]}')
            # unify urls
            url = normalize_url(url)

            if '/' in url:
                url = url.split('/')[0]

            # check the normalized url in whitelist
            is_white = is_domain_whitelisted(url.lower(), whitelist)

            # write the results to file
            fout.write('%s,%s,%0.4f,%0.4f\n' %(
                url, 
                'legit' if pred==0 or is_white else 'scam', 
                probs[0],
                probs[1],
            ))
            
            if(pred==0 or is_white):
                print(f'DOMAIN_PRED_CLASSIFIER:legit')
            else:
                print(f'DOMAIN_PRED_CLASSIFIER:scam')

            if pred == 1:
                scam_count += 1
            pred_url_labels[url] = pred.item()

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--input_file', type=str, help='input pickle files of features', required=True)
    parser.add_argument('--output_file', type=str, help='output file path', required=True)
    parser.add_argument('--threshold', type=float, help='classifier threshold', required=False, default=0.5)

    args = parser.parse_args()
    main(args)
