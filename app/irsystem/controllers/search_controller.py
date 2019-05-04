from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import numpy as np
import csv
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from nltk.tokenize import TreebankWordTokenizer
from sklearn import linear_model
import Levenshtein
# from wand.image import Image
# from wand.color import Color

apple_mult = 1.24
project_name = "sellPhones"
net_id = "Alvin Qu: aq38, Andrew Xu: ax28, Kevin Wang: kw534, Samuel Han: sh779"

@irsystem.route('/', methods=['GET'])
def search():
    check = False
    mate = False
    check2 = False
    mate2 = False
    past = request.args.get('past')
    past2 = request.args.get('past2')
    past3 = request.args.get('past3')
    past4 = request.args.get('past4')
    if past:
        check = True
        mate=True
    else:
        mate = True

    if past2 == "True":
        flag = True
    else:
        flag = False

    if past3:
        check2 = True
        mate2=True
    else:
        mate2 = True

    if past4 == "True":
        flag2 = True
    else:
        flag2 = False

    input_arr = []
    final = [[0],[0]]

    condition = request.args.get('condition')
    budget = request.args.get('budgets')
    preference = request.args.get('prefs')
    old_phone = ""
    old_phone = request.args.get('old_phone')
    feature_text = request.args.get('feature_text')

    print(condition,budget,preference,old_phone,feature_text)


    if not feature_text :
        return render_template('search.html', name=project_name,netid=net_id, check=check, check2=check2, mate2=mate2,  mate=mate, flag=flag, flag2 = flag2,
                                condition=condition, names=[], urls = [], budget=str(budget), features = "None", close_words=[],scores=[],price=[],spnames=[],
                                name_feat=[],name_val=[],spvals=[],prefs=str(preference),best_match="Not Found",old_phone="Not Found")

    if not old_phone:
        old_phone = "Not Found"
    if not condition:
        condition = "new"

    if not preference:
        preference = 0.40




    if budget and feature_text and condition:

        def main(budget, feature_text,condition,preference):
            feature_text = str(feature_text)
            preference = float(preference)
            phones = {}
            old_specs = {}
            labels = []
            with open('app/static/gsmphones.csv', mode='r',encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                for i,row in enumerate(csv_reader):
                    if i == 0:
                        labels = row
                    else:
                        phones[row[1]] = row[2:]
                        old_specs[row[1]] = row

            spec_to_index = {}
            for i in range(len(labels)):
                spec_to_index[labels[i].lower()] = i

            brand_dict = {}
            model_dict = {}
            for elt in old_specs:
                brand_dict[elt] = old_specs[elt][0].lower()
                model_dict[elt] = old_specs[elt][1].split(" ",1)[1].lower()

            dial_labels = ["announced","price","technology","weight","type","size","resolution","os",
               "cpu","internal","battery","single","3.5mm jack",
               "sensors","colors"]

            dial_to_output = ["Release","MSRP","Technology","Weight","Screen Type","Screen Size","Screen Resolution","OS",
               "Processor","Storage","Battery","Camera","Audio Jack",
               "Sensors","Colors"]

            nobat_phones = ["Oppo A5","i-mobile TV658 Touch&Move;","Nokia Lumia 900 AT&T;",
                "Sony Ericsson Jalou D&G; edition","AT&T; Quickfire",
                "AT&T; SMT5700","AT&T; Mustang","AT&T; 8525","HTC One X AT&T;"]

            with open('app/static/phones_dict.json','r',encoding='utf-8') as file:
                phones = json.load(file)
            with open('app/static/labels_dict.json','r',encoding='utf-8') as file:
                label_to_index = json.load(file)

            for elt in nobat_phones:
                if elt in old_specs:
                    old_specs.pop(elt)
                if elt in phones:
                    phones.pop(elt)

            rel_labels = [label_to_index["model image"],label_to_index["announced"],
                          label_to_index["cpu"],label_to_index["ppi"],label_to_index["storage"],
                          label_to_index["video"],label_to_index["dual"],label_to_index["ram"],
                          label_to_index["thickness"],label_to_index["sim"],label_to_index["card slot"],
                          label_to_index["rear camera"],label_to_index["front camera"],
                          label_to_index["size"],label_to_index["3.5mm jack"],label_to_index["waterproof"],
                          label_to_index["battery"],label_to_index["face"],label_to_index["finger"],
                          label_to_index["price"]]

            features = {}
            for i,phone in enumerate(phones):
                features[phone] = []
                for label in rel_labels:
                    features[phone].append(phones[phone][label])
                #features[phone] = np.array(features[phone])

            feat_to_index = {"announced":0,"cpu":1,"ppi":2,"storage":3,
                             "video":4,"dual":5,"ram":6,"thickness":7,"sim":8,
                             "card slot":9,"rear camera":10,"front camera":11,
                             "size":12,"3.5mm jack":13,"waterproof":14,
                             "battery":15,"face":16,"finger":17,"price":18}

            feature_mat = np.zeros((len(phones),len(features["Apple iPhone XS Max"])-1))

            phone_to_index = {}
            index_to_phone = []
            for i,phone in enumerate(features):
                feature_mat[i] = features[phone][1:]
                feature_mat[i,feat_to_index["thickness"]] = 20-feature_mat[i,feat_to_index["thickness"]]
                phone_to_index[phone] = i
                index_to_phone.append(phone)
            for i in range(len(feature_mat)):
                if feature_mat[i,feat_to_index["ram"]] > 4:
                    feature_mat[i,feat_to_index["ram"]] = 3
                if feature_mat[i,feat_to_index["storage"]] > 512:
                    feature_mat[i,feat_to_index["storage"]] = 64
            prices = feature_mat[:,feat_to_index["price"]]

            for i in range(len(feature_mat[0])):
                if i!=feat_to_index["price"]:
                    feature_mat[:,i] = feature_mat[:,i]/max(feature_mat[:,i])

            new_phones = {}
            new_prices = {}
            new_index  = {}
            old_phones = {}
            old_index  = {}
            for i in range(len(feature_mat)):
                if True not in np.isnan(feature_mat[i,:]):
                    if feature_mat[i,0]==1:
                        new_phones[index_to_phone[i]] = i
                        new_index[index_to_phone[i]]  = len(new_index)
                        new_prices[index_to_phone[i]] = feature_mat[i,feat_to_index["price"]]
                    else:
                        old_phones[index_to_phone[i]] = feature_mat[i,:]
                        old_index[len(old_index)]     = index_to_phone[i]

            phones_tr = []
            prices_tr = []
            new_index = {}
            for phone in new_phones:
                phone_vec = feature_mat[phone_to_index[phone],:]
                if True not in np.isnan(phone_vec):
                    new_index[len(phones_tr)] = phone
                    phones_tr.append(phone_vec[1:14])
                    prices_tr.append(new_prices[phone])
            phones_tr = np.array(phones_tr)
            prices_tr = np.array(prices_tr)

            phones_te = []
            for phone in old_phones:
                phone_vec = feature_mat[phone_to_index[phone],:]
                if True not in np.isnan(phone_vec):
                    phones_te.append(phone_vec[1:14])
            phones_te = np.array(phones_te)

            poly = PolynomialFeatures(degree=2)
            X_tr = poly.fit_transform(phones_tr)
            X_te = poly.fit_transform(phones_te)

            poly_model = linear_model.LinearRegression()
            poly_model.fit(X_tr, prices_tr)
            poly_pred  = poly_model.predict(X_te)

            new_ptoi = {}
            for phone in new_index:
                new_ptoi[new_index[phone]] = phone

            old_ptoi = {}
            for phone in old_index:
                old_ptoi[old_index[phone]] = phone

            def priceDiff(p,curr):
                return (phones[p][label_to_index["price"]]-curr)/phones[p][label_to_index["price"]]

            #NEW
            for p in old_ptoi:
                curr_price = feature_mat[phone_to_index[p]][feat_to_index["price"]]
                age = phones[p][label_to_index["age"]]

                curr_price *= 0.75
                curr_price -= 0.05*(1-max(0.5,priceDiff(p,poly_pred[old_ptoi[p]])))*age*curr_price
                feature_mat[phone_to_index[p]][feat_to_index["price"]] = curr_price

            #loading preprocessed review dictionaries
            with open('app/static/review_stuff.json', 'r',encoding='utf-8') as fp:
                review_stuff = json.load(fp)

            with open('app/static/sent_anal_dict.json', 'r',encoding='utf-8') as fp:
                sent_anal_dict = json.load(fp)

            with open('app/static/ratings.json', 'r',encoding='utf-8') as fp:
                ratings = json.load(fp)

            #creating vocab from reviews and list of phones that we have reviews for
            review_vocab = review_stuff['vocab']
            review_phonenames = review_stuff['phonenames']
            n_phones = len(review_phonenames)
            n_vocab  = len(review_vocab)

            #Create matrix from saves json of look-around sentiment analysis polarity values
            lookaround_matrix = np.zeros((n_phones,n_vocab))
            for key,value in sent_anal_dict.items():
                coord = key.split(",")
                x,y = int(coord[0]),int(coord[1])
                lookaround_matrix[x,y] = value

            def build_inv_idx(lst):
                """ Builds an inverted index.

                Params: {lst: List}
                Returns: Dict (an inverted index of phones)
                """
                inverted_idx = {}
                for idx in range(0,len(lst)):
                    inverted_idx[lst[idx]] = idx
                return inverted_idx

            #build inverted indexes for vocab and reviewed phone names
            review_vocab_invidx = build_inv_idx(review_vocab)
            review_names_invidx = build_inv_idx(review_phonenames)

            with open('app/static/word_to_index.json', 'r',encoding='utf-8') as fp:
                word_to_index = json.load(fp)

            index_to_word = {i:t for t,i in word_to_index.items()}

            words_compressed = np.load('app/static/words_compressed.npy')

            def closest_words(word_in, k = 10):
                if word_in not in word_to_index: return "Not in vocab."
                sims = words_compressed.dot(words_compressed[word_to_index[word_in],:])
                asort = np.argsort(-sims)[:k+1]
                return [(index_to_word[i],sims[i]/sims[asort[0]]) for i in asort[1:]]

            def query_word(word):
                close_words = closest_words(word)
                if close_words == "Not in vocab.":
                    return [word]
                return [word,close_words[0][0],close_words[1][0]]

            #USER INPUT
            input_query = feature_text

            #Processing user's input query
            treebank_tokenizer = TreebankWordTokenizer()
            input_query_clean = re.sub(r"[,;@#?!&$]+\ *", " ", input_query)
            input_query_token = treebank_tokenizer.tokenize(input_query_clean.lower())

            related_words1 = []
            related_words2 = []

            words_from_svd = []
            for word in input_query_token:
                sep = ", "
                svd_results = query_word(word)
                words_from_svd += svd_results
                if len(svd_results) > 1:
                    related_words2.append(svd_results[0] + " -> " + sep.join(svd_results[1:]))
                else:
                    related_words2.append(svd_results[0] + " -> ")

            svd_query = []
            rev_query = []
            for word in words_from_svd:
                svd_query.append(word)
                if word in review_vocab:
                    rev_query.append(word)

            feat_keywords = {'3.5mm jack': 'aux auxiliary 3.5mm 3.5 headphone jack port input',
                           'battery': 'battery life longlasting duration durable durability',
                           'card slot': 'sd slot micro microsd external',
                           'cpu': 'processor cpu speed lag power fast quick smooth performance chip chipset core quad octa hexa game games',
                           'dual': 'dualcameras dualcam zoom aperture',
                           'face': 'face unlock id facial recognition faceid security sensors',
                           'finger': 'fingerprint finger print sensor security sensors',
                           'front camera': 'front cameras selfie selfies snapchat instagram megapixel',
                           'ppi': 'screen resolution ppi dpi pixel pixels clarity hd amoled oled led',
                           'ram': 'memory ram lag',
                           'rear camera': 'rear camera cameras snapchat instagram megapixel picture pictures image images zoom bokeh',
                           'sim': 'sim dualsim simcard',
                           'size': 'screensize bezel big bigger huge giant large larger',
                           'storage': 'internal storage gb gigabytes',
                           'thickness': 'thinness thickness sleek slim thicc thick thin light lightweight weight weigh',
                           'video': 'video recording record videorecording stabilize stabilizer slowmo slow-mo motion fps',
                           'waterproof': 'water dust waterproof dustproof durable durability water/dust dust/water ip68 ip67',

                             #UNIMPLEMENTED
                            #'size2': 'small smaller light lightweight weight weigh pocketable pocketability',
                           #'fastcharge': 'quickcharge quickcharging fastcharge fastcharging charge charging turbocharge turbocharging',
                           #'protection': 'durable durability protection corning gorilla scratch drop shatter'
                          }

            feat_names = {'3.5mm jack': 'Audio Jack',
                           'battery': 'Battery Size',
                           'card slot': 'SD Card Slot',
                           'cpu': 'Processor (CPU)',
                           'dual': 'Dual Rear Cameras',
                           'face': 'Face Unlock',
                           'finger': 'Fingerprint Sensor',
                           'front camera': 'Camera (Front)',
                           'ppi': 'Screen Resolution',
                           'ram': 'Memory (RAM)',
                           'rear camera': 'Camera (Rear)',
                           'sim': 'Dual SIM',
                           'size': 'Screen Size (Bigger)',
                           'storage': 'Internal Storage',
                           'thickness': 'Thinness',
                           'video': 'Video Quality',
                           'waterproof': 'Water/Dust Resistance',

                          #UNIMPLEMENTED
                          #'size2': 'Size (Smaller)',
                           #'fastcharge': 'Fast/Quick Charging',
                           #'protection': 'Screen Protection (Shatter/Scratch-Resistant)'
                          }

            feat_names_inv = {i:t for t,i in feat_names.items()}


            def edit_distance_search(query, names):
                result = []
                for name in names:
                    score = Levenshtein.distance(query.lower(), name.lower())
                    result.append((score,name))
                result = sorted(result, key=lambda x: x[0])
                return result

            #fetching features
            feat_from_query = []
            for w in svd_query:
                for f in feat_keywords:
                    keywords = feat_keywords[f].split(" ")
                    for kw in keywords:
                        if w == kw:
                            feat_from_query.append(f)

            if len(feat_from_query) == 0:
                for w in svd_query:
                    topmatch = edit_distance_search(w, list(feat_names.values()))[0]
                    if topmatch[0] <= 7:
                        feat_from_query.append(feat_names_inv[topmatch[1]])

            feat_from_query = list(set(feat_from_query))

            for w in feat_from_query:
                if w not in rev_query and w in review_vocab:
                    related_words1.append(w + " (feature)")

            features_from_query = [feat_names[f] for f in feat_from_query]

            related_words = related_words1 + related_words2



            #RANKINGS using reviews
            n_words = len(rev_query)
            n_phones = len(review_phonenames)
            query_matrix = np.zeros((n_phones,n_words))

            for phone in review_phonenames:
                p = review_names_invidx[phone]
                for i,word in enumerate(rev_query):
                    w = review_vocab_invidx[word]
                    if i%3 != 0:
                        query_matrix[p,i] = lookaround_matrix[p,w] / 50
                    else:
                        query_matrix[p,i] = lookaround_matrix[p,w]

            #Outputting ranking based on social component
            query_matrix = np.sum(query_matrix, axis=1)
            query_matrix = query_matrix / len(rev_query)

            #WITH RATINGS (to be merged with cell above)
            for phone in review_phonenames:
                p = review_names_invidx[phone]
                rating = ratings[phone]
                rating_effect = 1.0
                ratio = rating/5.0
                polarity = query_matrix[p]

                if rating >= 4 and polarity > 0:
                    rating_effect = 2.0*ratio
                elif rating >= 4 and polarity < 0:
                    rating_effect = -1.5*ratio
                elif rating <= 2.5 and polarity > 0:
                    rating_effect = -1.5*ratio
                elif rating <= 2.5 and polarity < 0:
                    rating_effect = 2.0+(1-ratio)

                query_matrix[p] = rating_effect*polarity

            ranking_asc = list(np.argsort(query_matrix))
            ranking_desc = ranking_asc[::-1]

            # for i in range(30):
            #     print(str(i+1) + ". " + review_phonenames[ranking_desc[i]] + " -- "\
            #           + str(query_matrix[ranking_desc[i]] / query_matrix[ranking_desc[0]]))

            phone_to_review = {}
            for i in range(n_phones):
                phone_to_review[review_phonenames[ranking_desc[i]]] = query_matrix[ranking_desc[i]] / query_matrix[ranking_desc[0]]
            for phone in phones:
                if phone not in phone_to_review:
                    phone_to_review[phone] = 0


            bin_feats = {"3.5mm jack":0,"dual":1,"sim":2,"announced":3,"card slot":4,"waterproof":5,"face":6,"finger":7}

                    #Need user input
                    # min_price = 800
                    # max_price = 1600
                    # price_range = budget

            prices = feature_mat[:,feat_to_index["price"]]
            if int(budget) >= 600:
                starting = 200
            else:
                starting = 0
            ending = int(budget)
            price_range = np.intersect1d(np.where(prices>=starting)[0],np.where(prices<ending)[0])

            if condition == "new":
                condition = 1
            else:
                condition = 0
            query_feat = feat_from_query

            #query_feat = ["ram","front camera","cpu","rear camera"]
            # price_range= luxury

            if old_phone:
                old_query = old_phone
            else:
                old_query = ""


            def checkBinary(idx,query_feat):
                phone = index_to_phone[idx]
                for feat in query_feat:
                    if feat in bin_feats and feature_mat[idx][feat_to_index[feat]]!=1:
                        return False
                return True

            def getRank(dic):
                ranks = []
                for elt in dic:
                    ranks.append((elt,dic[elt]))
                ranks = sorted(ranks,key=lambda x: x[1])
                for elt in ranks:
                    print(elt)



            if len(query_feat)==0:
                query_feat = set(feat_to_index.keys())-set(bin_feats.keys())

            old_query = old_query.lower()
            old_brand = old_query.split(" ")[0]
            boosted_brand = ""
            matched_brand = edit_distance_search(old_brand,brand_dict.keys())
            if matched_brand[0][0] < 5:
                boosted_brand = matched_brand[0][1]

            results = {}
            ranked_results = []
            best_match = edit_distance_search(old_query,phones.keys())[0][1]
            best_dist  = edit_distance_search(old_query,phones.keys())[0][0]
            best_match_vec = feature_mat[phone_to_index[best_match]]
            best_vec = np.zeros(len(query_feat))


            physical_feat = True
            for elt in query_feat:
                if elt not in ["size","thickness"]:
                    physical_feat = False

            query_vec = np.zeros(len(query_feat))
            for i,feat in enumerate(query_feat):
                best_vec[i]  = best_match_vec[feat_to_index[feat]]
                query_vec[i] = max(feature_mat[:,feat_to_index[feat]])

            cossim = {}
            for p in price_range:
                if feature_mat[p][0] == condition and index_to_phone[p] != old_query and checkBinary(p,query_feat) \
                    and feature_mat[p][feat_to_index["thickness"]]>0.2 and feature_mat[p][feat_to_index["thickness"]]<1:
                    temp = np.zeros(len(query_feat))
                    for i,feat in enumerate(query_feat):
                        if feature_mat[p,feat_to_index[feat]] == 0:
                            temp[i] = 0.01
                        else:
                            temp[i] = feature_mat[p,feat_to_index[feat]]
                    if checkBinary(phone_to_index[best_match],query_feat):
                        cossim[p] = np.dot(temp,best_vec)/np.linalg.norm(temp)*np.linalg.norm(best_vec)
                    else:
                        cossim[p] = np.dot(temp,temp)/np.linalg.norm(temp)**2
            cossim_lst = []
            for idx in cossim:
                cossim_lst.append([idx,cossim[idx]])
            cossim_lst = np.array(cossim_lst)
            cossim_lst[:,1] /= max(cossim_lst[:,1])
            sim_index = np.argsort(cossim_lst[:,1])[::-1]
            cossim_min = min(cossim_lst[:,1])
            for idx in sim_index:
                results[index_to_phone[int(cossim_lst[idx][0])]] = (cossim_lst[idx,1]-cossim_min)/(max(cossim_lst[:,1])-cossim_min+1e-5)

            query_mat = []
            prange_to_phone = []
            for p in price_range:
                if feature_mat[p][0] == condition and index_to_phone[p] != old_query and checkBinary(p,query_feat) \
                    and feature_mat[p][feat_to_index["thickness"]]>0.2 and feature_mat[p][label_to_index["thickness"]]!=1:
                    prange_to_phone.append(index_to_phone[p])
                    query_mat.append(feature_mat[p][[feat_to_index[feat] for feat in query_feat]])
            query_mat = np.array(query_mat)

            cossim_vec = np.zeros(len(phones))
            for i,vec in enumerate(query_mat):
                vec /= query_vec
                brand = prange_to_phone[i].split(" ")[0]
                if not physical_feat:
                    if brand == "Apple" or brand == "Google" or brand == boosted_brand:
                        vec *= 1.2
                    elif brand == "Samsung":
                        vec *= 1.05
                cossim_vec[i] = np.linalg.norm(vec)
                cossim_vec[i] *= 1+0.125*phone_to_review[prange_to_phone[i]]

            rankings = np.where(cossim_vec>0)
            rankings = np.argsort(cossim_vec[rankings])[::-1]
            cossim_vec /= max(cossim_vec)
            found_phone = "none found"
            for idx in rankings:
                dist = Levenshtein.distance(best_match.lower(),prange_to_phone[idx].lower())
                if dist > 5:
                    results[prange_to_phone[idx]] += cossim_vec[idx]
                else:
                    results[prange_to_phone[idx]] += 0.75/(1-0.01*preference)*cossim_vec[idx]
            if best_dist > 5:
                best_match = "Not Found"

            final_rank = []
            for phone in results:
                if not physical_feat:
                    final_rank.append((phone,results[phone]*(1+min(2,1+feature_mat[phone_to_index[phone]][feat_to_index["price"]]/1000))))
                else:
                    final_rank.append((phone,results[phone]))
            final_rank = sorted(final_rank,key=lambda x: x[1])[::-1]
            result = []
            urls   = []
            for elt in final_rank:
                result.append(elt[0])
                urls.append(features[elt[0]][0])

            scores = []
            ml_prices = []
            feat_plot = []
            feat_val = []
            spec_names = dial_to_output
            spec_vals = []
            for i in range(len(result)):
                scores.append(np.round(final_rank[i][1],2))
                ml_prices.append(np.round(feature_mat[phone_to_index[result[i]]][feat_to_index["price"]],2))
                spec_vals.append([])
                for j,feat in enumerate(dial_labels):
                    spec_vals[i].append(old_specs[result[i]][spec_to_index[feat]])
                # specs[i] = []
                # for j,feat in enumerate(dial_labels):
                #     specs[i].append(old_specs[result[i]][spec_to_index[feat]])

            for i,elt in enumerate(spec_vals):
                if len(spec_vals)>72:
                    spec_vals[i] = spec_vals[i][:72]

            for i in range(len(scores)):
                scores[i] /= int(np.max(scores))+1
                scores[i] = np.round(scores[i],3)

            feat_labels = {'3.5mm jack': 'Audio Jack',
                       'battery': 'Battery Size',
                       'card slot': 'SD Card Slot',
                       'cpu': 'Processor (CPU)',
                       'dual': 'Dual Rear Cameras',
                       'face': 'Face Unlock',
                       'finger': 'Fingerprint Sensor',
                       'front camera': 'Front Camera',
                       'ppi': 'Screen Resolution',
                       'ram': 'Memory (RAM)',
                       'rear camera': 'Rear Camera',
                       'sim': 'Dual SIM',
                       'size': 'Screen Size (Bigger)',
                       'storage': 'Internal Storage',
                       'thickness': 'Thinness',
                       'video': 'Video Quality',
                       'waterproof': 'Water/Dust Resistance',
                      }


            def plot_bar(phone, features, i):
                label = []
                feat_scores = []
                for f in features:
                    label.append(feat_labels[f])
                    feat_scores.append(np.round(feature_mat[phone_to_index[phone]][feat_to_index[f]],2))
                return label,feat_scores

            for i,phone in enumerate(result[:18]):
                feat_plot,feat_val = plot_bar(phone,query_feat,i)

            # def clean_phone(url, i):
            #     with Image(filename=url) as img:
            #         img.format = 'png'
            #         with Color('#FDFDFD') as white:
            #             twenty_percent = int(65535 * 0.02)
            #             img.transparent_color(white, alpha=0.0, fuzz=twenty_percent)
            #         img.save(filename="clean_phone_%i.png" % (i+1)) #CHANGE TO DYNAMIC NAME OF PHONE RANKING

            # for i,phone in enumerate(result[:18]):
            #     clean_phone(features[phone][0], i)

            return [result,urls,related_words,scores,ml_prices,spec_names,feat_plot,feat_val,spec_vals, best_match]

        final = main(budget,feature_text,condition,preference)


        return render_template('search.html', name=project_name,netid=net_id, check=check, check2=check2, mate2=mate2, mate=mate, flag=flag, flag2=flag2,
                            condition=condition, names=final[0], urls = final[1],budget=str(budget),close_words = final[2],scores=final[3],
                            price=final[4],spnames=final[5],name_feat=final[6],name_val=final[7],spvals=final[8], prefs=str(preference), features=feature_text,
                            best_match=final[9],old_phone = old_phone)
