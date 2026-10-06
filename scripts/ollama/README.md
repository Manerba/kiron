# Isolierter Ollama-CPU-Vertragstest

`smoke-runtime.py` erstellt ausschließlich neue Testreports unter
`/usr/lib/kiron/test-runtimes/ollama/reports/cpu-NAME`. Es installiert nichts,
zieht keine Images/Modelle und kontaktiert keine produktive Ollama-API.
Der native Lauf ist erst nach unabhängiger Prüfung und einer unmittelbar
vorhergehenden Betriebsbestandsaufnahme durch Root vorgesehen. Parallel zu
Prism-/Embedding-Modelltests wird er nicht gestartet.

Der Pin ist das bereits vorhandene lokale Image
`sha256:684edc911db13b64ad072af7f83c3ca2e6545e4933961d67ed11f2893f897525`
(Ollama 0.18.0), mit geprüftem RepoDigest und vorhandenem Compat-Report.
Der qwen3:8b-Manifestdigest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
bindet alle Modell-, Template-, Parameter- und Lizenzblobs. Vor dem Lauf und
nach dem Cleanup werden ihre Größen/Hashes geprüft. Keine Modellkopie und kein
Host-Ollama-Binary werden verwendet.

Vorbereitung und späterer, separat geprüfter Start in einem neuen Reportpfad
(`cpu-contract-v10` ist bereits abgeschlossen und bleibt unverändert):

```sh
/usr/lib/kiron/test-venvs/local-inference/bin/python -I -B \
  /opt/kiron/scripts/ollama/smoke-runtime.py prepare \
  --report-dir /usr/lib/kiron/test-runtimes/ollama/reports/cpu-contract-NEXT

/usr/lib/kiron/test-venvs/local-inference/bin/python -I -B \
  /usr/lib/kiron/test-runtimes/ollama/reports/cpu-contract-NEXT/source/scripts/ollama/smoke-runtime.py run \
  --report-dir /usr/lib/kiron/test-runtimes/ollama/reports/cpu-contract-NEXT
```

`cpu-contract-v1` bleibt als fehlgeschlagener Versuch vor dem Containerstart
unverändert archiviert. Sein Dockerfehler enthielt noch kein stderr. Neue Läufe
halten Dockeroperation, Exitcode bzw. Timeout und höchstens 4096 Zeichen
bereinigtes stderr im Fehlerbericht fest; Zugangsdaten und Terminalsteuerzeichen
werden entfernt. Aus dem v1-Bericht allein ist die Fehlerursache nicht belegt.
`cpu-contract-v2` belegt die konkrete Docker-Ablehnung von `rprivate` bei einer
Quelle innerhalb des Daemon-Verzeichnisses `/var/lib/docker`; auch dieses Archiv
bleibt unverändert. Der feste Mount nutzt nun `readonly,bind-propagation=rslave`:
Mountänderungen propagieren nur vom Host zum Container. Es gibt keinen Fallback.
Bei der lesenden Prüfung am 22.09.2026 lagen unter der Cachequelle keine
Submounts. Der Preflight lehnt vorhandene Cache-Submounts ausdrücklich ab:
DockerCLI erlaubt `bind-recursive=readonly` nur mit `rprivate`, das der Daemon
hier wiederum verbietet. Diese unzulässige Kombination wird nicht verwendet.
Vor SDK-Start und bei jeder Prozessmessung müssen `/models` und sämtliche dort
sichtbaren Submounts in `/proc/PID/mountinfo` tatsächlich `ro` sein. Nachträgliche
Host-Mountänderungen werden damit beobachtet, nicht atomar ausgeschlossen.
Die Einweg-Propagation entspricht dem [Docker-Mountvertrag](https://docs.docker.com/engine/storage/bind-mounts/#configure-bind-propagation).

Root kontrolliert den eigenen Docker-Container. SDK-Prozess und nativer Server
laufen als UID65534/GID982 mit NoNewPrivs. Der SDK-Prozess hat eine leere
zusätzliche Gruppenliste. Der gemessene Docker/runc-Vertrag wiederholt die
primäre GID982 als einzige Gruppe: exakt `[982]`, keine weitere GID. Die
Nativeprüfung und Metriken verwenden diese tatsächlichen Prozesswerte;
abweichende UID/GID/Groups/NoNewPrivs/CapEff werden begrenzt diagnostiziert.
`runc`, keine DeviceRequests/Devices, keine Capabilities, schreibgeschütztes
Root-Dateisystem und schreibgeschützter Cache verhindern GPU-/Cache-Mutationen.
Nur `/tmp` ist als 64MiB-tmpfs beschreibbar. Docker begrenzt den Container auf
zwei CPU-Kerne, 9GiB RAM, keinen zusätzlichen Swap und 128 Prozesse. Effektive
cgroup-Werte, UID/GID/Groups/NoNewPrivs und fehlende GPU-Geräte werden geprüft.
Jede Messung hält die tatsächlichen Strings aus `cpu.max`, `memory.max`,
`memory.swap.max` und `pids.max` fest. `max`, falsche numerische Grenzen und
Lesefehler brechen ab; die Diagnose enthält PID, Cgroup und alle vier Werte
(je höchstens 128 Zeichen). Es wird kein Limit auf dem Host verändert.
Der Server lauscht ausschließlich auf `127.0.0.1:18095`; der feste Host-Netzmodus
ist keine Netzsperre. Es gibt keine programmierte Pull-/Cloud-/Produktivroute.

Die unveränderliche Sourcekopie und `plan.json` binden den Entwicklungsstand.
Dateimenge, Rechte und Hashes der Sourcekopie werden vor dem Start und nach dem
Cleanup erneut geprüft; `source_verified_after` hält den Nachbeleg fest.
Registry und Admission sind isolierte Fixtures. Der kontrollierte erste native
CPU-Load nutzt Kontext1024 und num_gpu0; Threads/Batch bleiben auf den Defaults
des gepinnten Images, identisch zu späteren Adapteranfragen. Die cgroup begrenzt
die gesamte CPU-Zeit des Containers auf zwei Kerne. Erst nach exaktem
`/api/ps`-Nachweis werden bestehende Residency und gemessene Modellgröße im
Test-Admissionstore registriert. Der Resolver hat ausdrücklich **kein**
ResourceProfile: Ein später nötiger Coldload durch RuntimeService ist dadurch
geschlossen. Das ist kein Nachweis einer produktiven Coldload-Ressourcenfreigabe.

Nach CAP-02 bindet die vorbereitende CHAT-Capability das beobachtbare
Residentprofil ausdrücklich: `context_tokens=(1024,)`, `device=('cpu',)`.
Der echte Provider prüft frische `/api/ps`-Residency unmittelbar vor jedem
Inferenz-POST und setzt `num_ctx=1024`, `num_gpu=0`, `truncate=false`,
`shift=false`. Der Probe injiziert weder `options_policy` noch `keep_alive`;
nur der separate kontrollierte Preload verwendet zehn Minuten Keep-alive.
Die Implementation wird von Anfang an wie produktiv aus dem exakten
`compat.image_digest`-RepoDigest, `template_revision=None` und der aktuellen
Adapterrevision gebaut. Der Compatdigest muss einem exakt geplanten RepoDigest
entsprechen. Image-ID/RootFS und Modell-/Templateblobs bleiben eigene
Archivpins, ersetzen aber nicht diese Implementationidentität.

Danach laufen echter OpenAI-SDK2.29.0, echte ASGI-API, RuntimeService und
OllamaProvider: Text JSON/SSE, automatische Einzel-/Paralleltools, Replay mit
umgekehrter Ergebnisreihenfolge und unterschiedlichen ALPHA/BETA-Werten, JSON
Object und striktes JSON Schema. Native Request-/Responsebodies und EOF sowie
SDK-Ergebnisse bleiben im Report. Aktives Reasoning, Bilder, required/named und
strikte Tools werden vor nativer I/O abgewiesen. Kandidatenfähigkeiten gelten
nur für diesen Test; daraus wird nichts veröffentlicht.

Die bestandene und unabhängig geprüfte v9-Probe ergänzt G1 mit vier tatsächlichen SDK-Anfragen:

| Ergebnisdatei | Eingabe | Geforderter öffentlicher Abschluss |
|---|---|---|
| `sdk-context-single-JSON.json` | Eine lange Usernachricht | HTTP 400, `context_length_exceeded` |
| `sdk-context-single-SSE.json` | Dieselbe lange Usernachricht | SDK-SSE-Fehler `context_length_exceeded` |
| `sdk-context-history-JSON.json` | 32 ältere abwechselnde User-/Assistantnachrichten und kurze letzte Usernachricht | HTTP 400, `context_length_exceeded` |
| `sdk-context-history-SSE.json` | Derselbe vollständige Verlauf | SDK-SSE-Fehler `context_length_exceeded` |

Die festen Eingaben sind begrenzt (je unter 32 KiB). Ihre Länge wird nicht
heuristisch in Tokens umgerechnet: Nur die tatsächliche native Ablehnung
belegt die Kontextüberschreitung. Die Probe verlangt bytegetreu erhaltene
Nachrichten im nativen JSON, genau einen Chat-POST und unmittelbar davor eine
erfolgreiche CPU-/Kontextbeobachtung. Der zugeordnete native Trace enthält
Request, Status, unveränderte Fehlerbytes und EOF. Erlaubt sind der exakte
native HTTP-400-Fehler oder bei Streaming dessen vollständiger einzelner
NDJSON-Fehlerframe mit Status 400. Öffentliche Content-/Tooldeltas, Usage oder
Erfolgsfinish sind verboten; ein initialer Assistant-Rollenchunk ist zulässig.
Nach bestätigter Ablehnung darf kein Requestticket verbleiben. Stille
Trunkierung, falsche Fehlerklassen oder unbekanntes Ende lassen den Gate
scheitern. `sdk-context-recovery.json` belegt anschließend eine kurze normale
Antwort. Damit werden acht positive, vier Kontextfehler und die bisherigen
fünf No-I/O-Ablehnungen geprüft. Im Lauf vom 22.09.2026 lieferten alle vier
Kontextfälle native HTTP-400-Antworten mit sauberem EOF und anschließend
jeweils null Requesttickets. Die Recovery lieferte 20 Input- und drei
Outputtokens. Container-, Prozess-, Cgroup-, Port- und Storecleanup sind
bestätigt; die Produktionsbeobachtungen unterscheiden sich ausschließlich im
Erhebungszeitpunkt. Der externe unabhängige Bericht steht unter
`/usr/lib/kiron/test-runtimes/review-candidates/tareas-60-remediation-20260922/ollama-cpu-v9-independent-review.json`.

`cpu-contract-v5` bleibt ein fehlgeschlagener nativer Weather-Roundtrip:
Trotz korrekt übergebener ALPHA/BETA-Ergebnisse erzeugte das Modell
`Berlin=22;Paris=20`; die öffentliche Antwort entsprach exakt der nativen.
Das gepinnte Template rendert die Toolhistorie auch bei `tools:[]`.
Die v6-Fixture verwendet deshalb die sachlich passende Funktion
`lookup_value(city)` mit JSON-Ergebnissen `{"value":"ALPHA"}` bzw.
`{"value":"BETA"}`. Ergebnisse enthalten keine Stadt, kommen in umgekehrter
Callreihenfolge an und werden nur über die Call-ID zugeordnet. Der Userprompt
nennt ausschließlich die Ausgabeform, keine erwarteten Werte. Die strikte
Zuordnungsprüfung bleibt bestehen; weder die Fixtureänderung noch v5 erteilen
eine native Ollama-Roundtripfreigabe.

`cpu-contract-v7` bleibt fehlgeschlagen: Nach drei korrekten Cgroupmessungen
wurde vor den SDK-Fällen ein `max` statt einer numerischen Speichergrenze
beobachtet. Der alte Fehlertext trennt `memory.max` und `memory.swap.max` nicht.
Im Journal ist unmittelbar vorher eine durch `apt-daily-upgrade.service`
angeforderte systemd-Neuladung um 04:08:01–02 UTC belegt; die zeitliche Korrelation
beweist keine konkrete Ursache der Limitänderung. CPU-Load war erfolgreich,
Cleanup vollständig. Neue Läufe diagnostizieren die vier Limitdateien einzeln;
v7 erteilt keinen Featuregrant.

Der SDK-`.stream()`-Helper akzeptiert Tools nur mit `strict=True` zur automatischen
Argumentdekodierung. Da Ollama diese Kontrolle nicht belegt, verwendet die
Toolstream-Probe das öffentliche `create(stream=True)` und den gepinnten
`ChatCompletionStreamState` ohne Auto-Parsing. Der vollständige Assistantdump
wird anschließend unverändert wieder eingelesen. Text und Schema verwenden den
normalen SDK-Helper. Es wird keine strikte Toolfähigkeit behauptet.

Der Workflow hat 600s Budget; Setup und Unload kommen hinzu. Der Root-Supervisor
bricht nach 900s, `STOP` im Report oder unter 2GiB verfügbarem Host-RAM ab.
Die CPU/RAM-cgroup-Grenzen gelten für den nativen Container; die RAM-Schwelle
ist zusätzlich eine Mess-/Abbruchgrenze des gesamten Hosts. CLI- und Cleanup-
Wartezeiten sind separat begrenzt. Cleanup stoppt/entfernt ausschließlich die
vollständig verifizierte Container-ID mit passendem Runlabel. Bei unbestätigtem
Cleanup ist der gesamte Report fehlgeschlagen; Produktionsbelegung wird nie
automatisch verändert. Die Root-Betriebsbestandsaufnahme vor/nach dem Lauf bleibt
separat erforderlich, da dieser Teststore keine Produktivkonkurrenz sperrt.

Offlineprüfungen ohne Docker-/Modellstart:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=services/kiron-common:services/kiron-proxy \
  /usr/lib/kiron/test-venvs/kitt-worker/bin/python -m pytest scripts/ollama/test_smoke_runtime.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=services/kiron-common:services/kiron-proxy \
  /usr/lib/kiron/test-venvs/local-inference/bin/python -m unittest discover \
  -s services/kiron-proxy/contracts -p test_ollama_probe_sdk.py -q
```

## Historischer Reviewkandidat aus v8

`cpu-contract-v8` ist nativ bestanden und unabhängig nachgeprüft: sieben
positive SDK-/Nativefälle, fünf Ablehnungen ohne native Anfrage, exaktes
ALPHA/BETA-Replay und vollständiges Container-/Admissioncleanup. Die 110
Sourcepins, Image-/Cachebytes und alle 72 nativen Status-/EOF-Belege stimmen.
Die Produktionsfelder sind vor/nach identisch; die Nachbeobachtung erfolgte
bewusst verzögert, etwa vier Minuten nach Containerende. Alle 53 Messpunkte
bestätigen die effektiven Limits 2 CPU-Kerne, 9 GiB RAM, Swap 0 und 128 PIDs;
Peak 5425,023 MiB, verfügbarer Host-RAM mindestens 15604,887 MiB. Der ältere bestandene v6-Lauf bleibt
historisch erhalten; sein Sourcepin enthält den später korrigierten
Per-Tool-Strict-Vertrag noch nicht und wird nicht exportiert.

Der spätere #296-Review fand zusätzlich CAP-02 und G1. V8 belegt deshalb
keine behobene Residencybindung oder Trunkierungsabwehr. Sein Archiv und
Kandidat bleiben unverändert historisch; für die korrigierte Implementation
sind neue native Ergebnisse und eine gesondert geprüfte Exportrezeptur nötig.

Die ursprüngliche v8-Rekonstruktionsrezeptur war ausschließlich an dieses
Archiv gebunden. Sein unabhängig geprüfter Gesamtinventar-SHA256 ist
`fd859b1e9c279f031ca19a7afaf43b0f4e6f4a60ce9b2d158da19c2c4d51fbf7`.
Andere, veränderte oder durch neueren Runtimecode überholte Belege werden
abgewiesen; es gibt keinen frei übergebbaren Evidencehash oder Archivpfad.
Das Werkzeug liest Modell-/Cachedateien erneut, startet aber weder Docker noch
einen Provider und führt keinen Archivcode aus.

Die historische Ausgabe enthält `ollama.json` im bestehenden Evidenceformat und
`provenance.json`, root:GID982, Verzeichnis0750/Dateien0640. Der enge Satz umfasst
user/assistant/tool, Temperatur0, `max_completion_tokens` mit Budgets8/24/96
(Default8 als ausdrückliche lokale Policy), automatische/nonstrikte Tools und
`none`, höchstens eine Definition, Einzel-/Parallelcalls sowie JSON Object und
die Schemafeatures des ausgeführten festen strikten Objektschemas. Systemrolle,
aktive Reasoningausgabe, Vision, Responses, Cached-/Reasoningzählung, Embeddings,
GPU und Coldload bleiben ohne positiven Grant.
Gemessen wurden genau zwei parallele Calls. Die boolesche Parallelfähigkeit
enthält keine zusätzliche Maximalzahl; die normale Codecgrenze von 64 Calls
bleibt eine aus der allgemeinen Implementierung abgeleitete Grenze, kein
nativer Nachweis für 64 Modellcalls.

Die historische v8-Evidence bleibt an Image-ID `684edc…` **und** Template-SHA `ae370d…` gebunden.
Produktiv bindet die Composition RepoDigest `e23e…` und gegebenenfalls
Template=None, weil der Manifestdigest bereits das Template bindet. Diese
verschiedenen Identitäten werden nicht umgeschrieben. Der Reviewkandidat ist
kein unmittelbar installierbarer Produktionsgrant; die separat autorisierte
Abnahme einer endgültigen Betriebsidentität bleibt erforderlich.

## Historischer Reviewkandidat aus v9

Die damalige feste Rezeptur band ausschließlich das Archiv
`cpu-contract-v9` an seinen unabhängig geprüften Gesamtinventar-SHA256
`c4c94756bac6319e7789ed8d8e704408f65aae8ceb305a162f470029645caca7`
(691 Einträge). Das Archiv und der Kandidat bleiben unverändert erhalten;
die aktuelle v10-Rezeptur akzeptiert sie nicht. Die v9-Evidence verwendet die gemessene
Produktionsform `RepoDigest/Template=None` und trägt verpflichtend
`context_tokens=(1024,)`, `device=('cpu',)`. Ein GPUresident oder ein anderer
Kontext wird damit weder öffentlich angeboten noch zur Inferenz zugelassen.

Der Umfang der positiven Fähigkeiten bleibt eng wie oben beschrieben; hinzu
kommen der echte Kontextablehnungsnachweis und die anschließende Recovery.
`provenance.json` enthält beide Fallgruppen und den vollständigen Archivindex.
Ausgabe: `ollama.json` und `provenance.json`, root:GID982, Verzeichnis0750,
Dateien0640. Es erfolgt keine Installation in ein produktives Evidencedirectory.

## Aktueller Reviewkandidat aus v10

Die feste Rezeptur bindet ausschließlich `cpu-contract-v10` an den unabhängig
geprüften Gesamtinventar-SHA256
`ddf07bd850b39c17f2f9cd570d50f1844dc896c3c452d0e458dd9a799c250418`
(692 Einträge einschließlich Verzeichnissen). Result-SHA256:
`190881acf586e018b9478c49b74d3910bee8f86ee1cf07c4c7ef016e4ed591a4`.
Der unabhängige Bericht liegt unter
`/usr/lib/kiron/test-runtimes/review-candidates/tareas-60-remediation2-20260922/ollama-cpu-v10-independent-review.json`
(SHA256 `65bc7965a6abc939bc7fccf6cb6d9f62e53148a853fcfd8ad383dddbb11e1c8d`).

Der neue Lauf bindet 111 aktuelle Runtimequellen einschließlich der korrigierten
Ollama-Lifecyclekoordination. Zwölf native Chat-POSTs belegen acht positive Fälle
(sieben Basisfälle plus Recovery) und vier Kontextablehnungen mit sauberem EOF
und ohne verbliebene Requesttickets; fünf weitere Fälle werden vor nativer I/O
abgewiesen. Der unabhängige Peer bestätigt Cache-/Image-RootFS-Pins, effektive
CPU-/RAM-/Swap-/PID-Grenzen, Cleanup und Produktionsgleichheit bis auf den
Erhebungszeitpunkt. Die Capabilitygrenzen bleiben unverändert: CPU1024,
resident-only ohne Coldloadbudget, RepoDigest/Template=None, eng belegte
Text-/Tool-/Schemaoptionen. Eine positive Kandidatendatei ist keine
Produktionsfreigabe und kein Nachweis allgemeiner roher Artefaktverwaltung.

Die v10-Umstellung verändert ausschließlich die feste Exportbindung und diese
Dokumentation. Gemessene Runtime-/Adapterquellen, historische Archive und
bestehende Kandidaten werden nicht umgeschrieben. Ausgabe nur in ein neues
privates Ziel, root:GID982, Verzeichnis0750 und Dateien0640:

```sh
/usr/lib/kiron/test-venvs/local-inference/bin/python -I -B \
  scripts/ollama/export-review.py inspect
# Export nur in ein neues isoliertes Ziel; existierende Kandidaten bleiben unverändert.
/usr/lib/kiron/test-venvs/local-inference/bin/python -I -B \
  scripts/ollama/export-review.py export \
  --output /usr/lib/kiron/test-runtimes/ollama/capability-candidates/cpu-v10-review
/usr/lib/kiron/test-venvs/local-inference/bin/python -I -B \
  scripts/ollama/export-review.py verify \
  --candidate /usr/lib/kiron/test-runtimes/ollama/capability-candidates/cpu-v10-review
```

Rezepturtests (nur temporäre Dateien/Fakeprovider):
`PYTHONPATH=services/kiron-common:services/kiron-proxy
/usr/lib/kiron/test-venvs/kitt-worker/bin/python -m pytest
scripts/ollama/test_export_review.py -q`.
