import turicreate
import numpy as np

data = turicreate.SFrame('people_wiki.sframe')
print(data.column_names())

print(np.argwhere(data['name'] == 'Barack Obama'))
data['tfidf'] = turicreate.text_analytics.tf_idf(data['text'])
obama = data[data['name'] == 'Barack Obama']
clooney = data[data['name'] == 'George Clooney']
turicreate.distances.cosine(obama['tfidf'][0], clooney['tfidf'][0])

print(clooney.stack('tfidf',new_column_name=['word','tfs']).sort('tfs',ascending=False))

# model
model_test = turicreate.nearest_neighbor_classifier(data,features=['tfidf'],target='name')
# neighbor
knn_model = turicreate.nearest_neighbors.create(data,features=['tfidf'],label='name')

'''
note the difference!
'''

model_test.summary()

jolie = data[data['name'] == 'Angelina Jolie']
print(model_test.query(jolie))
turicreate.distances.cosine(jolie['tfidf'][0], clooney['tfidf'][0])

#cosine distance 0-1, the larger the more different.