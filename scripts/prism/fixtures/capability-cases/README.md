# Offline Fallfixtures

Gekürzte Daten aus den im jeweiligen JSON genannten isolierten Messarchiven.
Sie testen ausschließlich die sechs geschlossenen Fallvalidatoren. Modellstarts,
Produktionszugriff und externe Archive sind für diese Tests nicht erforderlich.

Die Fixtures enthalten keine vollständigen Inventare/Registry-/Bundlepins und
keine frische Sourceidentität. Sie sind **niemals exportierbare Evidence**.
Historische Fallformen bleiben absichtlich als Parserregression erhalten; der
separate Exportpfad lehnt historische Archive ab. Native verbose-Daten wurden
auf die für originale Token-ID-/Zählregeltests benötigten Felder gekürzt.

`public-responses.json` ergänzt für RR-02 den ausdrücklich synthetischen Fall
`tools-ordered-replay`: geordnete `[call,text,call]`-History und umgekehrte,
verschiedene Antworten. Antwort und Usage sind aus dem früheren einfachen Replay
kopierte Testdaten, keine Messung dieses neuen Inputs. Der Fall prüft ausschließlich
die geschlossene Validatorform; ein neuer erfolgreicher nativer Lauf ist für neue
Evidence nötig. Die ursprünglichen Laufarchive bleiben unverändert.
