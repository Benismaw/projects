#!/bin/bash
#fichier d'entrée
fichier_entree="$1"

#Vérification de l'argument
if [ -z "$fichier_entree" ]; then
    echo "Usage: $0 fichier.ass"
    exit 1
fi

#Extraction du nom et du dossier
dir="$(dirname "$fichier_entree")"
base="$(basename "$fichier_entree" .ass)"
elf="$dir/$base.elf"

#Compilation
echo "Compilation ARM (gcc) de $fichier_entree ..."
arm-none-linux-gnueabihf-gcc -static -x assembler "$fichier_entree" -o "${fichier_entree%.ass}.elf"

#verify errors during compilation
if [ $? -ne 0 ]; then
    echo ""
    echo "Erreur lors de la compilation"
    exit 1
fi

#Execution
echo "Execution de $elf:"
echo ""
qemu-arm "$elf"

#Suppression des fichiers temporaires
rm "${fichier_entree%.ass}.elf"