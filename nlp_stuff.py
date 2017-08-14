import spacy
import ipdb
import string

class QueryParser(object):
	"""
	Collection of utils for parsing the query.
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

# nlp = spacy.load("en")

# ## Get spacy doc.
# text = "cooking pizza in a pan"
# doc = nlp(unicode(text))

# ## Get the root.
# root = [w for w in doc if w.head is w][0]

# ## Collapse the phrases into single entities.
# for _nc in doc.noun_chunks:
# 	print _nc, _nc.root, _nc.root.tag