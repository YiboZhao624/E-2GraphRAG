
# retriever should be init with cache_tree and G : nx.netxxx
# also index.

class Retriever:
    def __init__(self) -> None:
        self.cache_tree = None
        self.G = None
        self.index = None
        pass