I want to achieve a multi-agent system which process the following task:
 - The program receive a query in natrual language from the user. The query will help the user to re-rank their predicted high-confidence alleles/genes/drugs in a context of tissue/types of a given diseases.
 - The program should provide a simple webpage frontend, where the user can input the query and visualize the program's output. 
 - The program is based on LLM and access the LLM through OpenAI compatible API. The webpage of the program also has two input box, one for base_url of the API and one for the key of the API.
 - By using LLM, the program isolates the alleles/genes/drugs into a list.
 - For each item in the list, the program construct a item's query consists of the item, the tissue/types and the disease from the user's query, then the item's query will be queried through API defined in src/pubvec/core/api.py for articles in PubMed. The model will be set to DeepSeek for the parameters of the API. The API will return a summary about the efficacy of the item against the particular tissue/types of the disease.
 - The summary for all items in the list will be gathered, and fed to the LLM, tasking the LLM to rank all items for their efficacy.
 - The rank will be visualized on the webpage.
