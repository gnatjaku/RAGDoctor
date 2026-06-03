# RAGDoctor

RAGDoctor to edukacyjny kontekst systemu odpowiadania na pytania na podstawie dokumentow medycznych. Celem kontekstu jest precyzyjne rozroznianie pojec zwiazanych z RAG i asysta medyczna.

## Language

**RAGDoctor**:
Edukacyjny prototyp systemu RAG dla dokumentow medycznych, ktory odpowiada na pytania na podstawie wczesniej przygotowanego kontekstu. Nie jest narzedziem diagnostycznym ani medycznym systemem decyzyjnym.
_Avoid_: Lekarz AI, system diagnostyczny, medyczny chatbot diagnostyczny

**Lokalny backend AI**:
Tryb pracy, w ktorym generowanie odpowiedzi i/lub embeddingow odbywa sie lokalnie, bez wysylania zapytan do zewnetrznego API modelu.
_Avoid_: Offline diagnostyka, prywatny lekarz AI

**Magazyn chunkow i embeddingow**:
Miejsce przechowywania fragmentow dokumentow, ich reprezentacji wektorowych oraz metadanych zrodel, wykorzystywane pozniej do wyszukiwania kontekstu dla pytania.
_Avoid_: Baza diagnoz, medyczna baza wiedzy

**Asystent pytaniowy nad dokumentami medycznymi**:
Interfejs do zadawania pytan o tresc wczesniej przygotowanych dokumentow medycznych. Nie oznacza asystenta klinicznego ani systemu wspierania decyzji medycznych.
_Avoid_: Asystent diagnostyczny, lekarz AI, kliniczny asystent medyczny
