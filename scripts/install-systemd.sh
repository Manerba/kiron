#!/bin/bash
set -e

echo "Installiere Kiron systemd-Units..."
cp /opt/kiron/systemd/kiron-proxy.service /etc/systemd/system/
cp /opt/kiron/systemd/kiron-docling.service /etc/systemd/system/
cp /opt/kiron/systemd/kiron-embeddings.service /etc/systemd/system/
cp /opt/kiron/systemd/kiron-deberta.service /etc/systemd/system/
systemctl daemon-reload

# #509: Drift-Check — warnen wenn alte kiron-*.service Dateien im
# systemd-Verzeichnis liegen, fuer die es keine Quelle mehr in
# /opt/kiron/systemd/ gibt (z.B. nach Service-Umbenennung). Kein
# automatisches rm wegen Risiko bei Fehl-Match; Cleanup ist Operator-
# Entscheidung.
drift=0
for installed in /etc/systemd/system/kiron-*.service; do
    [ -e "$installed" ] || continue
    name=$(basename "$installed")
    if [ ! -f "/opt/kiron/systemd/$name" ]; then
        echo "WARN: $installed hat keine Quelle in /opt/kiron/systemd/ — manuell pruefen/entfernen"
        drift=1
    fi
done
if [ "$drift" = "1" ]; then
    echo "Hinweis: nach manuellem rm bitte 'systemctl daemon-reload' ausfuehren."
fi

echo "Systemd-Units installiert. Aktivieren mit:"
echo "  systemctl enable --now kiron-proxy kiron-docling kiron-embeddings kiron-deberta"
