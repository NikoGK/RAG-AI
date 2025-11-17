import os
from flask import Flask, render_template, request, jsonify
from rag_core import index_document_chunk, retrieve_and_generate_answer
from PyPDF2 import PdfReader

app = Flask(__name__)

# --- Forside Route ---
@app.route('/')
def index():
    # Renderer din index.html fil i templates mappen
    return render_template('index.html') 

# --- Hjælpefunktion til PDF-ekstraktion ---
def extract_text_from_pdf(file_stream):
    """Bruger PyPDF2 til at udtrække tekst fra en PDF-filestream."""
    text = ""
    try:
        reader = PdfReader(file_stream)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Fejl ved PDF-ekstraktion: {e}")
        return "" # Returner tom streng ved fejl

# --- API til Fil-Upload (Indeksering) ---
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Ingen fil valgt'}), 400

    file = request.files['file']
    file_name = file.filename
    
    if file_name == '':
        return jsonify({'error': 'Tomt filnavn'}), 400
    
    try:
        file.seek(0)
        
        if file_name.lower().endswith('.pdf'):
            # Håndter PDF-filer
            cleaned_content = extract_text_from_pdf(file)
            
            if not cleaned_content:
                raise ValueError("Kunne ikke udtrække læsbar tekst fra PDF'en.")

        else:
            # Håndter standard tekstfiler (txt, csv, md)
            file_content_bytes = file.read()
            file_content = file_content_bytes.decode('utf-8', errors='ignore')
            cleaned_content = file_content.replace('\x00', '')
        
        # Kald kernefunktionen med det rensede indhold.
        # (index_document_chunk vil derefter chunk/embed det rene indhold)
        response_data = index_document_chunk(cleaned_content, file_name)

        return jsonify({"message": f"Dokument '{file_name}' indekseret succesfuldt.", "data": response_data}), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Fejl ved upload/indeksering: {e}") 
        error_msg = str(e.json().get('message', str(e))) if hasattr(e, 'json') else str(e)
        return jsonify({'error': f'Intern serverfejl under indeksering. Detalje: {error_msg}'}), 500

# --- API til Spørgsmål/Svar (Query) ---
@app.route('/api/query', methods=['POST'])
def rag_query():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'Mangler spørgsmål (query)'}), 400

    try:
        # Kald kernefunktionen fra rag_core.py
        answer = retrieve_and_generate_answer(query)
        
        return jsonify({'answer': answer}), 200

    except Exception as e:
        print(f"Fejl ved query: {e}") # Denne linje viser detaljerne i din terminal
    # Returner fejldetaljen til browseren for at identificere problemet
        return jsonify({'error': f'Intern serverfejl under query. Detalje: {str(e)}'}), 500


    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Fejl ved query: {e}")
        return jsonify({'error': f'Intern serverfejl under query.'}), 500
    # app.py (MIDLERTIDIG DEBUG RETTELSE i rag_query ruten)

# --- Kør Flask App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

