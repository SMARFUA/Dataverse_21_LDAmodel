import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

def compute_optimize3(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence and perplixity for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    perp_values = []
    model_list = []
    topics = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        perplex_model = model.log_perplexity(corpus,total_docs=len(texts))
        perp_values.append(perplex_model)
        topics.append(num_topics)

    return topics, coherence_values, perp_values

