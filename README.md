## This is the description of the main work during my research experience at Chinese Academy of Sciences.
### Background  
Our paper focused on developing an AI agent that can be used in company qualification evaluation processes.  
To put it specifically, the ultimate goal of the agent is to streamline the process of evaluating the qualification of upstream raw material supply companies, giving out the evaluation results in different levels after we pass it some data and information of any company.

### Technical Methods and Improvements
With the rapid development of LLMs, we decided to use LLMs, treating them as assistants in our project.

However, it is widely known that the hallucination of the responses LLMs generate remains as one of the biggest problems that are dragging us from giving total trust to AI answers.

To solve this problem, we turned to RAG(Retrieval Augmented Generation) technique, which significantly enlarges the knowledge base of LLMs in a specific area. In this case, this area is apparently the company qualification evaluation work.
That's exactly what we do in this part: We transmit a series of rating results of human experts based on hundreds of indicators that cover the top-five important aspects of company qualification status.

### Results
With the use of RAG technique, the accuracy of our prediction rose to 87.5% from 65.1%, which is exactly the number when only using LLMs to finish the work.

The improvement of the accuracy proves the value of our work. Many thanks and heartfelt gratitude to my professor and other research assistants as well!
