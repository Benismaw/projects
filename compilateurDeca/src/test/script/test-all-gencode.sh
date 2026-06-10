#!/bin/bash

# Se placer à la racine du projet
cd "$(dirname "$0")"/../../.. || exit 1

PATH=./src/test/script/launchers:./src/main/bin:"$PATH"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NO_COLOR='\033[0m'

passed_tests=0
failed_tests=0

# --- Pour les tests qui DOIVENT marcher parfaitement ---
test_gencode_valid() {
    local test_file="$1"
    local asm_file="${test_file%.deca}.ass"
    local res_file="${test_file%.deca}.res"

    rm -f "$asm_file" 2>/dev/null

    if ! decac "$test_file" > /dev/null 2>&1; then
        echo -e "${RED}[FAIL] Compilation échouée:${NO_COLOR} $test_file"
        failed_tests=$((failed_tests + 1))
        return 1
    fi

    output=$(ima "$asm_file" 2>&1)
    if [ $? -ne 0 ]; then
        echo -e "${RED}[FAIL] Runtime error inattendu:${NO_COLOR} $test_file"
        echo "Sortie : $output"
        failed_tests=$((failed_tests + 1))
        rm -f "$asm_file"
        return 1
    fi

    if [ -f "$res_file" ]; then
        if ! diff -u <(echo "$output") "$res_file" > /dev/null; then
            echo -e "${RED}[FAIL] Sortie incorrecte:${NO_COLOR} $test_file"
            echo "---- obtenu ----"
            echo "$output"
            echo "---- attendu ----"
            cat "$res_file"
            echo "----------------"
            failed_tests=$((failed_tests + 1))
            rm -f "$asm_file"
            return 1
        fi
    fi

    echo -e "${GREEN}[OK]${NO_COLOR} $test_file"
    passed_tests=$((passed_tests + 1))
    rm -f "$asm_file"
}

# --- Pour les tests qui DOIVENT compiler mais échouer à l'exécution ---
test_gencode_invalid_runtime() {
    local test_file="$1"
    local asm_file="${test_file%.deca}.ass"

    rm -f "$asm_file" 2>/dev/null

    # 1. La compilation DOIT réussir
    if ! decac "$test_file" > /dev/null 2>&1; then
        echo -e "${RED}[FAIL] La compilation aurait dû réussir:${NO_COLOR} $test_file"
        failed_tests=$((failed_tests + 1))
        return 1
    fi

    # 2. L'exécution DOIT échouer
    output=$(ima "$asm_file" 2>&1)
    if [ $? -eq 0 ]; then
        echo -e "${RED}[FAIL] L'exécution aurait dû échouer (erreur runtime attendue):${NO_COLOR} $test_file"
        echo "Sortie obtenue (pas d'erreur) : $output"
        failed_tests=$((failed_tests + 1))
        rm -f "$asm_file"
        return 1
    fi

    echo -e "${GREEN}[OK]${NO_COLOR} $test_file (Erreur runtime détectée)"
    passed_tests=$((passed_tests + 1))
    rm -f "$asm_file"
}

# --- Lancement des tests ---

echo "----------------------------------------"
echo -e "${BLUE}LANCEMENT DES TESTS GENCODE RECURSIFS${NO_COLOR}"
echo "----------------------------------------"

# 1. Tests Valides
echo -e "\n${YELLOW}>>> Tests Valides${NO_COLOR}"
while read -r test_file; do
    test_gencode_valid "$test_file"
done < <(find src/test/deca/codegen/valid -type f -name "*.deca" | sort)

# 2. Tests Performance
echo -e "\n${YELLOW}>>> Tests Performance${NO_COLOR}"
if [ -d "src/test/deca/codegen/perf" ]; then
    while read -r test_file; do
        test_gencode_valid "$test_file"
    done < <(find src/test/deca/codegen/perf -type f -name "*.deca" | sort)
fi

# 3. Tests Invalides (Runtime)
echo -e "\n${YELLOW}>>> Tests Invalides (Doivent échouer à l'exécution)${NO_COLOR}"
while read -r test_file; do
    test_gencode_invalid_runtime "$test_file"
done < <(find src/test/deca/codegen/invalid -type f -name "*.deca" | sort)

echo " "
echo "----------------------------------------"
echo -e "Total tests exécutés : $((passed_tests + failed_tests))"
echo -e "Tests REUSSIS        : ${GREEN}$passed_tests${NO_COLOR}"
echo -e "Tests ECHOUES        : ${RED}$failed_tests${NO_COLOR}"
echo "----------------------------------------"

if [ "$failed_tests" -eq 0 ]; then
    exit 0
else
    exit 1
fi