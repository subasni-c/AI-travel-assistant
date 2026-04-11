from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

from src.retriever import retrieve_docs
from src.config import MODEL_PROVIDER, OPENAI_API_KEY

# ── 1. Build the LLM ─────────────────────────────────────
if MODEL_PROVIDER == "openai":
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
else:
    raise ValueError(f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}")

# ── 2. Per-session memory store ───────────────────────────
_memory_store: dict[str, ConversationBufferMemory] = {}

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in _memory_store:
        _memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    return _memory_store[session_id]

def clear_memory(session_id: str) -> None:
    if session_id in _memory_store:
        del _memory_store[session_id]

# ── 3. Language instruction (REMOVED AUTO DETECT) ─────────
def get_language_instruction(language: str) -> str:
    """Returns language instruction for the selected language"""
    return (
        f"\n\nLANGUAGE INSTRUCTION:\n"
        f"You MUST respond ENTIRELY in {language}.\n"
        f"Every word of your answer must be in {language}.\n"
        f"Do not mix any other language.\n"
        f"Even if the question is in a different language, answer in {language}."
    )

# ── 4. Query rewriter ─────────────────────────────────────
REWRITE_SYSTEM_PROMPT = """You are a query rewriter for a travel assistant.
Your ONLY job is to rewrite the user's latest question so it is completely
self-contained and explicit — replacing all vague pronouns and references
(like "there", "it", "that place", "both", "the first one", "that city")
with the actual destination names found in the conversation history.
Rules:
- If the question mentions multiple destinations implicitly, name ALL of them.
- If the question is already explicit and clear, return it unchanged.
- Return ONLY the rewritten question. No explanation. No extra text.
- Do not answer the question. Just rewrite it.
- Always rewrite in English regardless of input language.
Conversation history:
{chat_history}
"""
REWRITE_HUMAN_PROMPT = "Rewrite this question to be explicit: {question}"

rewrite_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(REWRITE_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(REWRITE_HUMAN_PROMPT)
])

if MODEL_PROVIDER == "openai":
    rewrite_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0)
else:
    rewrite_llm = llm

def rewrite_query(raw_query: str, chat_history_text: str) -> str:
    if not chat_history_text.strip():
        return raw_query
    vague_words = [
        "there", "it", "that place", "both", "the city",
        "that country", "first one", "second one", "those",
        "the destination", "that", "here", "same place"
    ]
    if not any(word in raw_query.lower() for word in vague_words):
        print(f"[QueryRewriter] Already explicit — skipping: '{raw_query}'")
        return raw_query
    print(f"[QueryRewriter] Rewriting vague query: '{raw_query}'")
    formatted = rewrite_prompt.format_messages(
        chat_history=chat_history_text,
        question=raw_query
    )
    result    = rewrite_llm.invoke(formatted)
    rewritten = result.content.strip() if MODEL_PROVIDER == "openai" else str(result).strip()
    print(f"[QueryRewriter] Result: '{rewritten}'")
    return rewritten

# ── 5. Prompts (REMOVED MORNING/AFTERNOON/EVENING FORMAT) ─
PDF_ANSWER_SYSTEM_PROMPT = """You are an expert and friendly AI Travel Assistant.
Use the following travel guide context to answer the user's question.

CRITICAL RULES:
1. Read the user's question and identify what destination they are asking about
2. Read the context carefully
3. Ask yourself: "Does this context contain useful travel information about the destination the user asked about?"
4. If YES → answer from context normally
5. If NO, or if context is about a completely different place → respond with exactly: NO_PDF_CONTEXT

IMPORTANT RULES FOR NO_PDF_CONTEXT:
- Return ONLY the single word: NO_PDF_CONTEXT
- No emoji, no explanation, no extra text — just: NO_PDF_CONTEXT
- Do this ONLY when context has ZERO relevant info about what user asked
- If context has even partial info about user's destination → use it and answer

═══════════════════════════════════════════════════════════════
ANSWER FORMAT INSTRUCTIONS:
═══════════════════════════════════════════════════════════════

STEP 1: IDENTIFY QUERY TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━
Detect what user is asking:
- Full trip/plan → Give day-by-day itinerary in NATURAL, USER-FRIENDLY format
- Hotels         → List hotels only
- Prices/cost    → Price breakdown only
- Food           → Restaurants/cuisine only
- Beaches        → Beach list only
- Activities     → Activity list only

STEP 2: STRICT ANSWER RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOLDEN RULE: Answer EXACTLY what user asked. Nothing more. Nothing less.

IF user asks "plan a trip"         → give ONLY day-by-day itinerary
IF user asks "hotels"              → give ONLY hotels
IF user asks "food"                → give ONLY food/restaurants
IF user asks "beaches"             → give ONLY beaches
IF user asks "trip with hotels"    → give itinerary AND hotels
IF user asks "trip with price"     → give itinerary AND price

WARNINGS — STRICT RULES:
- ONLY add ⚠️ warning if user SPECIFICALLY asked for that info AND it is missing from context
- If user asked ONLY "plan a trip" → NO hotel warning, NO price warning, NOTHING extra
- If user asked "hotels" and hotels NOT in context → add warning
- If user asked "price" and price NOT in context → add warning
- If user did NOT ask for something → DO NOT mention it at all

STEP 3: FORMAT BEAUTIFULLY (USER-FRIENDLY!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOR TRIP ITINERARIES - NATURAL, CONVERSATIONAL FORMAT:
✅ Use simple day headers like "📅 Day 1", "📅 Day 2"
✅ Use bullet points (-) for each activity
✅ Describe what to do in flowing sentences
✅ Include 3-5 activities per day with descriptions
✅ Make it engaging and easy to read
✅ Add blank line between days
✅ End with "Enjoy your trip to [DESTINATION]! 🎉"

FOR OTHER QUERY TYPES:
- Hotels: List with categories (Luxury / Mid-range / Budget)
- Food: Categorize by cuisine type
- Activities: Group by theme
- Beaches: List with brief descriptions

STRICTLY FORBIDDEN:
❌ NEVER use "Morning:", "Afternoon:", "Evening:" labels
❌ NEVER give one-sentence-only activities
❌ NEVER say "NO_PDF_CONTEXT" when you have context about the destination
❌ Write naturally like you're helping a friend plan their trip

═══════════════════════════════════════════════════════════════

Context from travel guides:
{{context}}

Conversation so far:
{{chat_history}}

{language_instruction}


IMPORTANT: If the context above contains information about the destination in the question, USE IT to create a helpful answer. Only respond "NO_PDF_CONTEXT" if the context is truly empty or about completely different destinations.
"""
GENERAL_ANSWER_SYSTEM_PROMPT = """You are an expert AI Travel Assistant.

⚠️ IMPORTANT: The user's question is NOT covered by uploaded PDF guides.
You are answering from your GENERAL TRAVEL KNOWLEDGE.

INSTRUCTIONS:
1. Provide helpful, accurate information
2. Format beautifully and naturally

FOR TRIP PLANS:
- Write in natural, flowing paragraphs
- Describe days conversationally
- Include 3-5 activities per day with details
- Make it engaging and easy to read
- NO "Morning/Afternoon/Evening" labels - just natural narrative

FOR OTHER QUERIES:
- Hotels: Categorized lists (Luxury / Mid-range / Budget)
- Food: Organized by cuisine type
- Activities: Grouped by theme
- Be friendly and helpful

Conversation so far:
{{chat_history}}

{language_instruction}
"""

# ── 6. Main answer function ───────────────────────────────
def generate_answer(
    query:       str,
    session_id:  str  = "default",
    use_general: bool = False,
    language:    str  = "English"  # No more "auto" option
) -> dict:
    """RAG pipeline with language support"""
    
    memory       = get_memory(session_id)
    history_vars = memory.load_memory_variables({})
    history_msgs = history_vars.get("chat_history", [])

    history_text = ""
    for msg in history_msgs:
        role          = "User" if msg.type == "human" else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    # Translate non-English queries to English for search
    original_query = query
    
    if not all(ord(char) < 128 for char in query):
        print(f"[Generator] Non-English query detected: '{query}'")
        print(f"[Generator] Translating to English for search...")
        
        translate_prompt = f"Translate this to English, keep it concise: {query}"
        translation_response = llm.invoke([
            {"role": "system", "content": "You are a translator. Translate to English."},
            {"role": "user", "content": translate_prompt}
        ])
        query_for_search = translation_response.content.strip()
        print(f"[Generator] English translation: '{query_for_search}'")
    else:
        query_for_search = query

    rewritten_query = rewrite_query(query_for_search, history_text)
    lang_instruction = get_language_instruction(language)
    print(f"[Generator] Language: '{language}'")

    # ── Path A: General knowledge ─────────────────────────
    if use_general:
        print(f"[Generator] General knowledge path for: '{original_query}'")

        system_prompt = GENERAL_ANSWER_SYSTEM_PROMPT.format(
            language_instruction=lang_instruction
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("{question}")
        ])
        formatted = prompt.format_messages(
            chat_history=history_text,
            question=original_query
        )
        result = llm.invoke(formatted)
        answer = result.content.strip() if MODEL_PROVIDER == "openai" else str(result).strip()
        memory.save_context({"input": original_query}, {"answer": answer})
        return {
            "answer":          answer,
            "rewritten_query": rewritten_query,
            "has_pdf_context": False
        }

    # ── Path B: PDF retrieval ─────────────────────────────
    docs    = retrieve_docs(rewritten_query)
    context = "\n".join(docs)

    if not context.strip():
        print(f"[Generator] No PDF context for: '{original_query}' → asking user")
        return {
            "answer":          None,
            "rewritten_query": rewritten_query,
            "has_pdf_context": False
        }

    print(f"[Generator] PDF context found for: '{original_query}'")

    system_prompt = PDF_ANSWER_SYSTEM_PROMPT.format(
        language_instruction=lang_instruction
    )
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{question}")
    ])
    formatted = prompt.format_messages(
        context=context,
        chat_history=history_text,
        question=original_query
    )
    result = llm.invoke(formatted)
    answer = result.content.strip() if MODEL_PROVIDER == "openai" else str(result).strip()
    if "NO_PDF_CONTEXT" in answer:
        print(f"[Generator] LLM detected irrelevant context → asking user")
        return {
        "answer":          None,
        "rewritten_query": rewritten_query,
        "has_pdf_context": False
    }


    memory.save_context({"input": original_query}, {"answer": answer})
    return {
        "answer":          answer,
        "rewritten_query": rewritten_query,
        "has_pdf_context": True
    }