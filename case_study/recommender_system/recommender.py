import turicreate as tc
from collections import Counter
from collections import OrderedDict

song_data = tc.SFrame('song_data.sframe')
song_data.column_names()
song_count_id = Counter(song_data['song_id'])

sorted(song_count_id.items(), key=lambda x: x[1], reverse=True)


train, test = song_data.random_split(.8, seed=0)
popularity_model = tc.popularity_recommender.create(train, user_id='user_id', item_id='song')
unique_test_users = test['user_id'].unique()
popularity_model.recommend(users=unique_test_users[:1])
unique_train_users = train['user_id'].unique()
popularity_model.recommend(users=unique_train_users[:1])
# will be different from what the person previously had


# item similarity based

item_similarity_model = tc.item_similarity_recommender.create(train, user_id='user_id', item_id='song')
print(item_similarity_model.recommend(users= unique_test_users[:1]))
print(item_similarity_model.summary())
print(item_similarity_model.get_similar_items(['Secrets - OneRepublic']))
print(item_similarity_model.get_similar_items(['With Or Without You - U2']))

# model comparisons
model_performance = tc.recommender.util.compare_models(test, [popularity_model, item_similarity_model])
