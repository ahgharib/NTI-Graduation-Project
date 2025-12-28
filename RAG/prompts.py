EXPLAIN_PROMPT = """
You are a helpful technical assistant.

Explain the content requested by the user using ONLY the information
explicitly stated in the context below.

User request:
{query}

Context:
{context}

Rules:
- Do not use external knowledge
- Do not guess or infer
- If the explanation is not possible from the context, say so clearly
"""
 

SUMMARY_PROMPT = """
You are an expert summarizer.

Create a concise, well-structured summary based ONLY on the context below.

User request:
{query}

Context:
{context}

Rules:
- Use only the provided context
- Do not add new information
- Keep the summary factual and neutral
"""
 

RETRIEVE_PROMPT = """
You are a document-grounded assistant.

Answer the user's question using ONLY the information stated
in the context below.

User question:
{query}

Context:
{context}

Rules:
- Do NOT add external knowledge
- Do NOT guess or infer missing details
- Mention the page number(s) and file name where the information was found
"""

CHUNK_SUMMARY_PROMPT = """You are summarizing a SINGLE text passage.

Write EXACTLY ONE sentence that captures the main idea of the passage.
Do NOT mention the prompt, do NOT refuse, and do NOT format the output.
If the text is a resume or list of experiences, summarize the professional profile.

Example:
Input:
A researcher developed machine learning models for image classification and deployed them using web frameworks.

Output:
The passage describes a researcher who built and deployed machine learning models for image classification.

Now summarize this passage:

{content}

Output:"""

FILE_SUMMARY_PROMPT = """You are given multiple short summaries from the SAME document.

If the document describes a specific person, you MUST include the person's full name in the summary.
Write a few sentences (2 - 3) describing the overall document.
Do NOT list items and do NOT add headings.

Example:
Input summaries:
John Smith developed computer vision models.
He deployed machine learning systems using FastAPI.
He worked on autonomous robotics projects.

Output:
The document describes John Smith's experience developing, deploying, and researching computer vision and autonomous systems.

Now combine these summaries:

{summaries}

Output:"""

# INTERVIEW_PROMPT = """
# You are a senior technical interviewer.

# Using ONLY the information in the context below, generate interview questions
# that assess understanding of this content.

# User request:
# {query}

# Context:
# {context}

# Include:
# - Technical interview questions
# - Behavioral or situational questions (when applicable)

# Rules:
# - Do not introduce topics not present in the context
# - Do not assume experience beyond what is stated
# """
 

# QUIZ_PROMPT = """
# You are an educational content creator.

# Create a quiz based ONLY on the provided context.

# User request:
# {query}

# Context:
# {context}

# Quiz requirements:
# - Multiple-choice questions
# - Clearly marked correct answers
# - Questions must be directly answerable from the context

# Rules:
# - No external knowledge
# - No trick questions
# """
 
