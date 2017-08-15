import spacy
import ipdb
import string

class QueryParser(object):
	"""
	Collection of utils for parsing the query.
	Loading spaCy and the english language model for the
	dependency parser takes a while. Load it ONCE globally.
	"""
	def __init__(self):
		## Create spaCy model.
		self.nlp = spacy.load("en")
		self.whitelist = string.lowercase ## Confirm the whitelist!!

	def clean_string(self, qstring):
		qstring = qstring.strip().lower()
		return unicode(''.join([c if c in self.whitelist else ' ' for c in qstring]))

	def parse_the_string(self, uni_string):
		"""
		Extract and return noun-chunks, the sentence, the dependency phrases in a dict.
		"""
		assert isinstance(uni_string, unicode), "--The string is not unicode. Have you passed it through clean_string??--"

		doc = self.nlp(uni_string)
		root = [w for w in doc if w.head is w][0]
		noun_chunks = [tok.text for tok in doc.noun_chunks]

		## Merge chunks.
		for ph in doc.noun_chunks:
			ph.merge(ph.root.tag_, ph.text, ph.root.ent_type_)

		## Find all parent nodes.
		parents = [tok for tok in doc if len(list(tok.children))!=0]

		# ipdb.set_trace()

		


if __name__ == '__main__':
	QPObj = QueryParser()

	qString = "cooking pizza in a pan"
	cleanString = QPObj.clean_string(qString)
	QPObj.parse_the_string(cleanString)