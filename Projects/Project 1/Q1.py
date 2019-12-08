import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

data_set = [['M', 'O', 'N', 'K', 'E', 'Y'], ['D', 'O', 'N', 'K', 'E', '
Y'], ['M', 'A', 'K', 'E'], ['M', 'U', 'C', 'K', 'Y'], ['C', 'O', 'O', '
K', 'I', 'E']]

	def display(frame):
		print(f'\tTotal Relations found: {frame.shape[0]} with 60% minimum
		support and 80% minimum confidence\n')

		for i in range(frame.shape[0]):
			print(f'\tRelation: \t{list(frame.antecedents[i])} -> {list(frame.consequents[i])}')
			print(f'\tConfidence: {frame.confidence[i]}')
			print(f'\tSupport: \t{frame.support[i]}')
			print('\t--------------------------------')

		encoder = TransactionEncoder()
		fit_data_set = encoder.fit(data_set).transform(data_set)
		fit_data_set = pd.DataFrame(fit_data_set, columns=encoder.columns_)
		frame = fpgrowth(fit_data_set, use_colnames=True, min_support=0.6)
		frame = association_rules(frame, metric='confidence', min_threshold=0.8)

		print('Using FP-growth algorithm:')
		display(frame)
		frame = apriori(fit_data_set, use_colnames=True, min_support=0.6)
		frame = association_rules(frame, metric='confidence', min_threshold=0.8)

		print('Using Apriori algorithm:')
		display(frame)
