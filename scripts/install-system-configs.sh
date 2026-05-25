#!/bin/bash
set -e

echo "Installiere System-Konfigurationen..."

# NVIDIA unattended-upgrades Blacklist
if [ -f /opt/kiron/system/apt/50unattended-upgrades ]; then
    cp /opt/kiron/system/apt/50unattended-upgrades /etc/apt/apt.conf.d/50unattended-upgrades
    echo "  NVIDIA apt-Blacklist installiert"
fi

# nvidia-persistenced Drop-in
if [ -f /opt/kiron/system/systemd/nvidia-persistenced-enable.conf ]; then
    mkdir -p /etc/systemd/system/nvidia-persistenced.service.d/
    cp /opt/kiron/system/systemd/nvidia-persistenced-enable.conf \
       /etc/systemd/system/nvidia-persistenced.service.d/enable.conf
    systemctl daemon-reload
    if systemctl list-unit-files | grep -q '^nvidia-persistenced\.service'; then
        systemctl enable --now nvidia-persistenced.service
        echo "  nvidia-persistenced Drop-in installiert und aktiviert"
    else
        echo "  nvidia-persistenced Drop-in installiert, aber Unit nicht vorhanden (Paket fehlt) - enable uebersprungen"
    fi
fi

# Shared Runtime-Verzeichnis fuer kurzlebige GPU-Gate-Marker
if [ -f /opt/kiron/system/tmpfiles.d/kiron-runtime.conf ]; then
    mkdir -p /etc/tmpfiles.d
    cp /opt/kiron/system/tmpfiles.d/kiron-runtime.conf /etc/tmpfiles.d/kiron-runtime.conf
    systemd-tmpfiles --create /etc/tmpfiles.d/kiron-runtime.conf
    echo "  /run/kiron tmpfiles.d installiert"
fi

echo "Fertig."
