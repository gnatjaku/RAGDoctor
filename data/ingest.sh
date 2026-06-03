#!/usr/bin/env bash
set -euo pipefail

PAYLOAD=$(cat <<'EOF'
{
  "source_id": "treatment-001",
  "source_name": "treatment-flu.txt",
  "text": "Grypa jest ostrą chorobą zakaźną dróg oddechowych charakteryzującą się sezonowością występowania. Szczyt zachorowań w Polsce przypada na okres od stycznia do marca. Najbardziej skuteczną metodą zapobiegania grypie są szczepienia ochronne. Z artykułu dowiesz się, jak można odróżnić grypę od innych chorób układu oddechowego oraz jakie szczepionki obowiązują w sezonie 2024/2025. Grypa jest ostrą chorobą zakaźną wywoływaną przez wirusy z rodziny Orthomyxoviridae. Istnieje kilka typów wirusa, z czego dwa: A i B są przyczyną epidemicznych zachorowań. Wirus A jest chorobotwórczy dla ludzi i zwierząt i charakteryzuje się dużą zmiennością antygenową białek powierzchniowych – hemaglutyniny (H) i neuraminidazy (N). Grypę sezonową wywołują najczęściej wirusy podtypów H1N1 i H3N2 (w niektórych sezonach H1N2). Wirus B jest zakaźny wyłącznie dla ludzi, a objawy grypy przez niego wywołane są zazwyczaj łagodniejsze niż te powodowane przez wirusa A. Przypadki grypy występują w Polsce przez cały rok, ale sezon epidemicznych zachorowań przypada na okres od września do końca marca. Jak dochodzi do zakażenia grypą? Grypa jest chorobą wysoce zaraźliwą (jeden chory zaraża średnio 4 osoby z bliskiego otoczenia). Okres wylęgania objawów chorobowych jest krótki i wynosi zazwyczaj 1-2 dni. Do infekcji dochodzi: drogą kropelkową podczas kasłania, wydmuchiwania nosa, mówienia, drogą powietrzną, poprzez kontakt bezpośredni (styczność z osobą zakażoną), poprzez kontakt pośredni poprzez dotykanie skażonych powierzchni. Źródłem zakażenia są: dorosły na 1-2 dni przed pojawieniem się objawów, małe dziecko do 6 dni przed pojawieniem się objawów, chory dorosły do 5-7 dni po wystąpieniu objawów, chore dziecko przez okres powyżej 10 dni od wystąpieniu objawów, chory z ciężkim niedoborem odporności przez wiele tygodni lub miesięcy.",
  "metadata": {"category": "treatment"}
}
EOF
)

curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"
