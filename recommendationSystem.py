###########################################################
##   Author : Ajay Singh                               ####
#            -----------@gmail.com                      ### 
###########################################################

import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens



# Fetch data and format it
data = fetch_movielens(min_rating=4.0)

#print training and testing data
print(repr(data['train']))
# print(data['train'])
print(repr(data['test']))



# ---- Create Model
model = LightFM(loss='warp')
model_1 = LightFM(loss='logistic')
model_2 = LightFM(loss='bpr')
model_3 = LightFM(loss='warp-kos')

# --Train model
model.fit(data['train'], epochs=30, num_threads=2)
model_1.fit(data['train'], epochs=30, num_threads=2)
model_2.fit(data['train'], epochs=30, num_threads=2)
model_3.fit(data['train'], epochs=30, num_threads=2)



def sample_recommendation(model, data, user_ids):

    # number of users and movies in training data
    n_users, n_itmes, = data['train'].shape


    # generate recommendations for each user we input
    for user_id in user_ids:

        # movies which user already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts which user will like
        scores = model.predict(user_id, np.arange(n_itmes))
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_id)
        print("  Known Positives:--")

        for x in known_positives[:3]:
            print("   %s" % x)
        

        print("  Recommended --")

        for x in top_items[:3]:
            print("   %s" % x)


sample_recommendation(model, data, [3, 25, 450])
sample_recommendation(model_1, data, [3, 25, 450])
sample_recommendation(model_2, data, [3, 25, 450])
sample_recommendation(model_3, data, [3, 25, 450])