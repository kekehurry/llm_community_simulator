import faiss
import networkx as nx
import pickle
import numpy as np
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
import uuid
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
import re
import json
import pyvis.network as net

class AnswerOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        choice_pattern = re.compile(r'```(?:json)?(.*?)```', re.DOTALL)
        choice_match = choice_pattern.search(text)
        choice = ""
        try:
            choice = choice_match.group(1)
            return json.loads(choice.strip()), text.strip()
        except Exception as e:
            return {},text.strip()
        
class ChoiceModel:
    def __init__(self,
        graph_path="community_graph_embeded.pkl",
        embed_model="nomic-embed-text",
        chat_model="phi3:instruct",
        embedding_dimension=768,
        period=9
        ):
        self.embed_model = OllamaEmbeddings(model=embed_model)
        self.chat_model = ChatOllama(model=chat_model)
        self.embedding_dimension = embedding_dimension
        self._graph = self._load_graph(graph_path)
        self.graph = self._graph.copy()
        node_embeddings = np.array([node['embedding'] for idx,node in self._graph.nodes(data=True)])
        index = faiss.IndexFlatL2(self.embedding_dimension)
        index.add(node_embeddings)
        ids = np.array([node for node in self._graph.nodes()])
        self.index = {"index":index, "ids":ids}
        self.degree_centralities = self._cal_degree_centralities()
        self.period = period
        
    def _load_graph(self, graph_path):
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        return G
    
    def _load_index(self, index_path):
        with open("index.pkl", "rb") as f:
            index = pickle.load(f)
        return index
    
    def _cal_degree_centralities(self):
        degree_centralities = nx.degree_centrality(self._graph)
        return degree_centralities

    def get_period(self, period):
        related_links = [link for link in self._graph.edges() if self._graph.edges[link]["period"] == period]
        graph = self._graph.edge_subgraph(related_links).copy()
        return graph

    def roll_back(self, period):
        related_links = [link for link in self._graph.edges() if self._graph.edges[link]["period"] <= period]
        self.graph = self._graph.edge_subgraph(related_links).copy()
        self.period = period

    def similarity_search(self,query, k1=5, k2=2,node_type='Actors'):

        # step 1: find nodes with similar profile
        def semantic_similarity_search(query, k1):
            query = str(query)
            query_embed = self.embed_model.embed_query(query)
            query_embed = np.array(query_embed).reshape(1,-1)
            d, i = self.index["index"].search(query_embed, k1)
            similar_nodes = self.index["ids"][i][0]
            simiilar_scores = d[0]
            return similar_nodes,simiilar_scores

        # step 2: find nodes with similar degree_centrality in the community network
        def social_similarity_search(node_id,k2,node_type='Actors'):
            # filter all nodes with same type
            same_type_nodes = [node for node in self._graph.nodes() if self._graph.nodes[node]["type"] == node_type]
            node_degree_centrality = self.degree_centralities[node_id]
            node_degree_centralities = {k:abs(v-node_degree_centrality) for k, v in self.degree_centralities.items() if k in same_type_nodes}
            sorted_nodes = sorted(node_degree_centralities.items(), key=lambda x: x[1])
            sorted_similar_nodes = sorted_nodes[:k2]
            similar_nodes = [node for node, _ in sorted_similar_nodes]
            similar_scores = [score for _, score in sorted_similar_nodes]
            return similar_nodes,similar_scores

        # combine the two similarity search results
        nodes = []
        scores = []
        senmantic_similar_nodes,senmantic_similar_scores = semantic_similarity_search(query, k1)
        for node1,score1 in zip(senmantic_similar_nodes,senmantic_similar_scores):
            social_similar_nodes,social_similar_scores = social_similarity_search(node1,k2,node_type)
            filtered_nodes = [n for n in social_similar_nodes if n not in nodes]
            filtered_scores = [score1+score2 for score2,n in zip(social_similar_scores,social_similar_nodes) if n not in nodes]
            nodes.extend(filtered_nodes)
            scores.extend(filtered_scores)
        return nodes,scores

    def link_prediction(self, profile, k1, k2, node_type='Actors', choice_type='Actors', top_k=3):

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        def cal_weighted_jaccard_coeff(G,agent,choice):
            G_ = G.to_undirected()
            common_neighbors = list(nx.common_neighbors(G_,agent,choice))
            total_neighbors = len(list(nx.neighbors(G_,agent)))
            weight_sum = 0
            for n in common_neighbors:
                edge_ap = G_.edges[agent,n]['weight']
                edge_pc = G_.edges[n,choice]['weight']
                weight_sum += (edge_ap+edge_pc)/2
            return weight_sum/total_neighbors

        def get_recommendation(G,choice_type,top_k=None):
            agents = [n for n,d in G.nodes(data=True) if d['type'] == 'Agent']
            choices = [n for n,d in G.nodes(data=True) if d['type'] == choice_type]
            recommendation = {}
            for choice in choices:
                recommendation[choice] = cal_weighted_jaccard_coeff(G,agents[0],choice)
            recommendation = {k: v for k, v in sorted(recommendation.items(), key=lambda item: item[1],reverse=True)}
            choices = list(recommendation.keys())
            choices = [G.nodes[n] for n in choices]
            if top_k is None:
                top_k = len(choices)
            probabilities = softmax(list(recommendation.values())[:top_k]).tolist()
            return choices,probabilities

        # step 1: find nodes with similar profile
        similar_nodes,similar_scores = self.similarity_search(profile, k1, k2, node_type)

        # step 2: find neighbors of similar nodes
        neighbors = []
        for n in similar_nodes:
            neighbors.extend(list(self._graph.neighbors(n)))
            neighbors.append(n) 

        # step 3: create compute graph
        compute_graph = nx.subgraph_view(self._graph, filter_node=lambda n: n in neighbors).copy()
        # normalize the weights of links
        weights = nx.get_edge_attributes(compute_graph, 'weight')
        max_weight = max(weights.values())
        for e in compute_graph.edges():
            compute_graph[e[0]][e[1]]['weight'] = compute_graph[e[0]][e[1]]['weight'] / max_weight
        # normalize the weights of similarity scores
        max_score = max(similar_scores)
        similar_scores = [s / max_score for s in similar_scores]

        #step 4: add agent node
        agent_id = uuid.uuid4().hex
        compute_graph.add_node(
            agent_id, 
            type='Agent', 
            period = self.period, 
            properties=profile,
            label='Agent'
        )
        for node,scocre in zip(similar_nodes,similar_scores):
            compute_graph.add_edge(agent_id, node, weight=scocre, type='similar_to',label='similar_to') 
        
        # step 5: get recommendation
        choices,probabilities = get_recommendation(compute_graph,choice_type)

        return choices,probabilities
    
    def choose_from_link_prediction(self, profile, k1, k2, node_type='Actors', choice_type='Actors', top_k=3):
        choices,probabilities = self.link_prediction(profile, k1, k2, node_type, choice_type, top_k)
        choice = np.random.choice(choices, p=probabilities)
        return choice
    
    def get_old_context(self, profile, k1, k2, node_type='Actors', choice_type='Actors', top_k=3):
        choices,probabilities = self.link_prediction(profile, k1, k2, node_type, choice_type, top_k)
        options = [ c['properties'] for c in choices]
        options_probabilities = [ p for p in probabilities]
        old_context = {f"option:{o}, probability:{p}" for o,p in zip(options,options_probabilities)}
        return old_context
    
    def get_llm_choice(self,profile, new_options, node_type='Actors', choice_type='Actors', k1=10, k2=5, top_k=3):

        old_context = self.get_old_context(profile, k1, k2, node_type, choice_type, top_k)

        system_propmt = '''
        You are an expert analyst of social networks, given the context of connect propobilities with the similar node in the past, evaluate the possibilities of new options.
        '''
        user_prompt = '''
        Given a {node_type} Node with profile:
        {profile}
        And the context of {top_k} options and their probabilities a similar node connected in the past:
        {old_context}
        The new options are:
        {new_options}
        Is there any possibility for each new option the given node would connect to? 
        Note:
        - Answer with a list of 'Yes' or 'No' for each new option in json forma, using the index as the key for each option.
        - Wrap the final answer in triple backticks (```json ```) to indicate a json block.
        - Follw with the reasons for your choice after the json block.
        Answer Format Example:
        ```json
        {{
            "0": "Yes",
            "1": "No"
        }}
        [Reasons for the choices]
        ```
        '''

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_propmt),
                ("human", user_prompt),
            ]
        )
        choice_response = (
            RunnablePassthrough.assign(
                node_type=lambda _: node_type,
                profile=lambda _: profile,
                top_k=lambda _: top_k,
                old_context=lambda _: old_context,
                new_options=lambda _: new_options
            )
            | prompt
            | self.chat_model.bind()
            | AnswerOutputParser()
        )
        choice, response = choice_response.invoke({
            "node_type": node_type,
            "profile": profile,
            "top_k": top_k,
            "old_context": old_context,
            "new_context": new_options
        })
        return choice, response
    
    def evaluate(self,graph=None):
        if graph is None:
            graph = self.graph
        actors = [n for n in graph.nodes() if graph.nodes[n]['type'] == 'Actors']

        def cal_single_PPI(a):
            in_degree = graph.in_degree(a)
            out_degree = graph.out_degree(a)
            if (in_degree + out_degree) == 0:
                PPI_a = 0
            else:
                PPI_a = out_degree/(in_degree + out_degree)
            return PPI_a
    
        def cal_PPI():
            PPI = np.mean([cal_single_PPI(a) for a in actors])
            return PPI
        
        def cal_CVI():
            CVI = np.mean([graph.degree(a) for a in actors])
            return CVI
        
        def cal_DCI():
            degree_centrality = nx.degree_centrality(graph)
            actors_degree_centrality = [degree_centrality[a] for a in actors]
            M = np.mean(actors_degree_centrality)
            IQR = np.percentile(actors_degree_centrality, 75) - np.percentile(actors_degree_centrality, 25)
            DCI = IQR/M
            return DCI
        
        PPI = cal_PPI()
        CVI = cal_CVI()
        DCI = cal_DCI()
        
        return PPI, CVI, DCI
        
    
    def visualize(self, graph=None,file_name="visualize.html"):
        if graph is None:
            graph = self.graph
        visual_graph = graph.copy()
        color_map = {
        'Actors': '#00796B',
        'Media Content': '#EEFF41',
        'Organization': '#FBC02D',
        'Event': '#FF6E40',
        'Short-term Project': '#FF4081',
        'Long-term Project': '#2979FF',
        'Space': '#512DA8'
        # Add more types and colors as needed
        }
        for node in visual_graph.nodes():
            visual_graph.nodes[node]["color"] = color_map[visual_graph.nodes[node]["type"]]
        
        nt = net.Network("800px", "1200px")
        nt.from_nx(nx_graph=visual_graph)
        nt.write_html(file_name)
    


    