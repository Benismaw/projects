#!/bin/bash
cd "$(dirname "$0")"/../../.. || exit 1

PATH=./src/test/script/launchers:./src/main/bin:"$PATH"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NO_COLOR='\033[0m'

passed_tests=0
failed_tests=0

run_test() {
    local test_file="$1"
    local asm_file="${test_file%.deca}.ass"
    local res_file="${test_file%.deca}.res"

    rm -f "$asm_file" 2>/dev/null

    # Compilation
    if ! decac "$test_file" > /dev/null 2>&1; then
        echo -e "${RED}[FAIL] Compilation échouée:${NO_COLOR} $test_file"
        failed_tests=$((failed_tests + 1))
        return
    fi

    if [ ! -f "$asm_file" ]; then
        echo -e "${RED}[FAIL] Fichier .ass non généré:${NO_COLOR} $test_file"
        failed_tests=$((failed_tests + 1))
        return
    fi

    # Exécution
    output=$(ima "$asm_file" 2>&1)
    ima_exit_code=$?

    # Comparer avec le .res si présent
    if [ -f "$res_file" ]; then
        if ! diff -u <(echo "$output") "$res_file" > /dev/null; then
            echo -e "${RED}[FAIL] Sortie incorrecte:${NO_COLOR} $test_file"
            echo "---- obtenu ----"
            echo "$output"
            echo "---- attendu ----"
            cat "$res_file"
            failed_tests=$((failed_tests + 1))
            rm -f "$asm_file"
            return
        fi
    else
        # Aucun .res, afficher sortie mais ne pas compter comme fail
        echo -e "${YELLOW}[WARN] Fichier .res absent:${NO_COLOR} $test_file"
        echo "$output"
    fi

    echo -e "${GREEN}[OK]${NO_COLOR} $test_file"
    passed_tests=$((passed_tests + 1))
    rm -f "$asm_file"
}

echo "------------------------------"
echo -e "${BLUE}GENCODE TESTS - Avec Objet${NO_COLOR}"
echo "------------------------------"

# Valid tests récursifs
echo "--- Valid Tests ---"
while IFS= read -r test_file; do
    run_test "$test_file"
done < <(find src/test/deca/codegen/valid/avec_objet -type f -name "*.deca" | sort)

# Invalid tests récursifs
echo "   "
echo "--- Invalid Tests ---"
while IFS= read -r test_file; do
    run_test "$test_file"
done < <(find src/test/deca/codegen/invalid/avec_objet -type f -name "*.deca" | sort)

echo "------------------------------"
echo -e "Total tests : $((passed_tests + failed_tests))"
echo -e "Tests OK    : ${GREEN}$passed_tests${NO_COLOR}"
echo -e "Tests FAIL  : ${RED}$failed_tests${NO_COLOR}"
