#!/bin/bash

# Script d'installation de NaovaWebots
# Les fichiers doivent être téléchargés manuellement depuis :
# https://github.com/Naova/NaovaWebots/releases

set -e  # Arrête le script si une commande échoue

echo "=================================================="
echo "  Installation de NaovaWebots"
echo "=================================================="
echo ""

# Couleurs pour l'output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonctions pour afficher les messages
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérifier que unzip est installé
if ! command -v unzip &> /dev/null; then
    error "unzip n'est pas installé. Veuillez installer unzip :"
    echo "sudo apt install unzip"
    exit 1
fi

# Chercher webots_updated.zip dans ~/Downloads
DOWNLOAD_DIR="$HOME/Downloads"
if [ ! -f "$DOWNLOAD_DIR/webots_updated.zip" ]; then
    error "webots_updated.zip non trouvé dans $DOWNLOAD_DIR"
    echo ""
    echo "Téléchargez le fichier depuis :"
    echo "https://github.com/Naova/NaovaWebots/releases"
    echo ""
    echo "Le fichier doit être sauvegardé dans : $DOWNLOAD_DIR"
    exit 1
fi

# Étape 1: Décompresser webots_updated.zip
info "Étape 1/4: Décompression de webots_updated.zip..."
sudo unzip -q $DOWNLOAD_DIR/webots_updated.zip -d /usr/local/
if [ ! -d "/usr/local/webots" ]; then
    error "Erreur lors de la décompression de webots_updated.zip"
    exit 1
fi
info "Fichiers décompressés"

if [ -f "/usr/local/webots/webots" ]; then
    info "Fichier webots trouvé dans /usr/local/webots"
else
    error "Fichier webots non trouvé après décompression"
    exit 1
fi

# Étape 2: Configuration du .bashrc
info "Étape 2/4: Configuration de l'environnement (.bashrc)..."

if ! grep -q "WEBOTS_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Configuration Webots" >> ~/.bashrc
    echo "export WEBOTS_HOME='/usr/local/webots'" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$WEBOTS_HOME'/lib/controller'" >> ~/.bashrc
    info "Variables d'environnement ajoutées au .bashrc"
else
    warn "Variables d'environnement déjà présentes dans .bashrc"
fi

# Charger la nouvelle configuration
source ~/.bashrc

# Étape 3: Créer le lien symbolique
info "Étape 3/4: Création du lien symbolique..."

if [ -L "/usr/local/bin/webots" ]; then
    warn "Le lien symbolique /usr/local/bin/webots existe déjà"
else
    sudo ln -s /usr/local/webots/webots /usr/local/bin/webots
    info "Lien symbolique créé"
fi

# Vérifier que webots fonctionne
if command -v webots &> /dev/null; then
    info "Webots est accessible globalement"
else
    warn "Webots n'est pas accessible. Redémarrez votre terminal."
fi

# Étape 4: Installation des dépendances
info "Étape 4/4: Installation des dépendances..."

if [ -f "install_deps.sh" ]; then
    if [ ! -x "install_deps.sh" ]; then
        chmod +x install_deps.sh
    fi
    ./install_deps.sh
    info "Dépendances installées"
else
    error "Fichier install_deps.sh non trouvé"
    exit 1
fi

# Résumé final
echo "=================================================="
echo -e "${GREEN}Installation terminée avec succès!${NC}"
echo "=================================================="
echo ""
echo "Prochaines étapes :"
echo "1. Lancez Webots :"
echo "   webots"
echo ""
echo "2. Ouvrez le monde de simulation :"
echo "   File → Open World"
echo "   → k1_webots_simulation/K1_v1.wbt"
echo ""
echo "3. Pour plus d'informations, consultez le README.md"
echo ""
