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

    if ! decac -arm "$test_file" > /dev/null 2>&1; then
        echo -e "${RED}[FAIL] Compilation échouée:${NO_COLOR} $test_file"
        failed_tests=$((failed_tests + 1))
        return 1
    fi

    output=$(./arm.sh "$asm_file" 2>&1)
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
    if ! decac -arm "$test_file" > /dev/null 2>&1; then
        echo -e "${RED}[FAIL] La compilation aurait dû réussir:${NO_COLOR} $test_file"
        failed_tests=$((failed_tests + 1))
        return 1
    fi

    # 2. L'exécution DOIT échouer
    output=$(./arm.sh "$asm_file" 2>&1)
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

    
    echo "----------------------------------------"
    echo -e "${BLUE}TESTS GENCODE ARM${NO_COLOR}"
    echo "----------------------------------------"
}

# Test des fichiers valides
for test_file in src/test/deca/codegen/valid/provided/arithmetic/*.deca;
do
    test_gencode_valid "$test_file";
done

for test_file in src/test/deca/codegen/valid/sans_objet/*.deca;
do
    test_gencode_valid "$test_file";
done    


echo " "
echo "invalid tests:"
echo "--------------------------------"
# Test des fichiers invalides (runtime)

for test_file in src/test/deca/codegen/invalid/sans_objet/*.deca;
do
    test_gencode_invalid_runtime "$test_file";
done


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