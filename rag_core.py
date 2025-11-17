import os
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from urllib.parse import quote_plus

# Sørg for at miljøvariabler er indlæst
load_dotenv()

# --- Konfiguration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Modelkonstanter ---
# Denne embedding model er valgt for sin 768-dimensionelle vektor
EMBEDDING_MODEL = 'text-embedding-004' 
RAG_LLM_MODEL = 'gemini-2.5-flash'

# --- Initialisering ---
# Initialiser Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) 

# Initialiser Google AI - API-nøglen skal sendes direkte til Client
client = genai.Client(api_key=GEMINI_API_KEY)

# --- rag_core.py: get_google_embedding(text) ---
def get_google_embedding(text: str) -> list[float]:
    """
    Genererer embedding vektor for given tekst via Google Gemini API.
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text] 
    )
    
    # Hent Vektor: Sikker og robust adgang til de rå værdier
    try:
        # Prøver den mest moderne/almindelige adgang (.value)
        vector_data = response.embeddings[0].value 
    except AttributeError:
        # Hvis .value ikke virker, prøv .values
        try:
            vector_data = response.embeddings[0].values
        except AttributeError:
            # Hvis ingen af dem virker, antag at objektet allerede er selve vektoren
            vector_data = response.embeddings[0] 

    # SIKKERHED: Tving konverteringen til en ren Python list()
    if vector_data:
        embedding = list(vector_data)
    else:
        embedding = []

    return embedding

# --- Hjælpefunktion: Smartere text chunking ---
def smart_splitter(text: str, max_size: int = 1500, overlap: int = 200) -> list[str]:
    """
    Smartere chunking der prøver at bevare naturlige sektioner.
    Opdeler tekst på afsnit (dobbelt linjeskift) for bedre kontekst.
    """
    # Prøv først at splitte på dobbelt linjeskift (afsnit)
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # Hvis tilføjelse af dette afsnit ville overskride max_size
        if len(current_chunk) + len(para) > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start ny chunk med overlap
            current_chunk = current_chunk[-overlap:] + para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Tilføj sidste chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# --- Hovedfunktion: index_document_chunk ---
def index_document_chunk(file_content: str, file_name: str):
    """
    Håndterer chunking, embedding og lagring af et dokument i Supabase.
    """
    # 1. Tjek for tomt indhold
    if not file_content:
        raise ValueError("Filindhold er tomt.")
    
    # 2. Rens filnavn til at være URL-sikkert
    safe_file_name = quote_plus(file_name) 

    # 3. Opret chunks fra hele dokumentet med smartere chunking
    chunks = smart_splitter(file_content, max_size=2000, overlap=300)
    
    # 4. Indekser HVER chunk individuelt
    success_count = 0 
    for i, chunk in enumerate(chunks):
        
        # 4a. Generer Embedding
        embedding = get_google_embedding(chunk)
        
        # 4b. Gem metadata og embedding i 'documents' tabel
        data, count = supabase.from_('documents').insert([
            {
                "file_path": f"documents/{safe_file_name}_{i}", 
                "content": chunk,
                "embedding": embedding 
            }
        ]).execute()
        
        # Tæl op, hvis indsættelsen lykkedes
        success_count += 1
    
    # 5. Returner JSON-sikker bekræftelse
    return {"chunks_indexed": success_count, "total_chars": len(file_content)}

def retrieve_and_generate_answer(query: str) -> str:
    """ 
    Udfører den fulde RAG-cyklus: Retrieval og Generation.
    """
    # 1. Generer Embedding for Spørgsmålet
    query_embedding = get_google_embedding(query)
    
    # 2. Hent Kontekst fra Supabase (Vektorsøgning via RPC)
    documents_response = supabase.rpc('match_documents', { 
        'query_embedding': query_embedding,
        'match_count': 50, 
    }).execute()

    documents = documents_response.data
    context = "\n---\n".join([doc['content'] for doc in documents])
    
    if not context:
        return "Jeg har ingen relevant information i mine dokumenter til at svare på det spørgsmål."

    # DEBUG: Uncomment linjen nedenfor for at se den fundne kontekst
    # return context
    
    # 3. LLM Generation med forbedret system prompt
    system_prompt = f"""Du er en hjælpsom AI assistent. Brug den kontekst nedenfor til at svare på brugerens spørgsmål.

Vigtigt: Hvis du kun finder DELE af et svar (fx punkt 1 og 7 ud af en liste med 10 punkter), skal du:
- Angive de punkter du HAR fundet
- Tydeligt skrive hvilke punkter der mangler
- Ikke opfinde information om de manglende punkter

Kontekst:
---
{context}
---"""
    
    response = client.models.generate_content(
        model=RAG_LLM_MODEL,
        contents=[
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": query}]}
        ]
    )

    return response.text