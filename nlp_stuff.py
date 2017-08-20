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

		## For every parent, extract the dependency paths.
		node_paths = [(n, self.getPath(n, master=[], how="right")) for n in parents]

		## Now clean the paths - unwrap lists and get the text for tokens.
		node_paths = [( n, [QueryParser.unwrapList(path) for path in lol] ) for n,lol in node_paths]
		node_paths = [( n.text, [map(lambda x:x.text, l) for l in lol] ) for n,lol in node_paths]

		return {
		"root": root,
		"doc": doc,
		"noun_chunks": noun_chunks,
		"node_paths": node_paths
		}

		# print "--DEBUG--"
		# ipdb.set_trace()

	@staticmethod
	def unwrapList(some_list):
		"""
		Recursive function to unwrap nested lists.
		"""
		if not isinstance(some_list[0], list):
			return some_list
		return QueryParser.unwrapList(some_list[0])

	@staticmethod
	def getPath(node, master=[], how="right"):
		"""
		Recursive function to extract the dependency paths.
		"""
		assert how in ("right", "left"), "--How to add node is not clear.--"
		if how=="right":
			master.append(node)
		else:
			master = [node] + master
		
		if len(list(node.children))==0:
			return master
		
		lefts = list(node.lefts)
		rights = list(node.rights)

		## For every node in a parent's children, the next path has to be appened to a "new" master.
		## Hence the list(master) - this is to remove dependency from the existing master list.
		## i.e. for every new child, the existing master is "replicated" and used for independent dependency paths.
		return [QueryParser.getPath(child, list(master), "right" if child in rights else "left") for child in node.children]

if __name__ == '__main__':
	QPObj = QueryParser()

	qString = "cooking pizza in a pan"
	cleanString = QPObj.clean_string(qString)
	parse_dict = QPObj.parse_the_string(cleanString)

	print parse_dict
	ipdb.set_trace()