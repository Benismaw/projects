# Système de Gestion d'Épicerie

Application de gestion d'épicerie avec base de données relationnelle, développée en équipe.

## Ce que fait le projet

Permet de gérer les produits, stocks et commandes d'une épicerie via une interface Java connectée à une base Oracle.

- Modélisation relationnelle (SQL/Oracle)
- Transactions ACID
- Interface Java avec JDBC

## Tech Stack

SQL, Oracle, Java, JDBC

# Arborescence

<pre>
projet_bd/
|
+-- src/
|   +-- CommandeUtils.java
|   +-- Catalog.java
|   +-- ClientUtils.java
|   +-- StockUtils.java
|   +-- PasserCommande.java
|   +-- PasserCommandeEnPrete.java
|   +-- GererAlertesPeremption.java
|   +-- CloturerCommande.java
|   +-- QueryInput.java
|   +-- LireInput.java
|   +-- Main.java
|
+-- sql/
|   +-- create_tables.sql
|   +-- insert_test_data.sql
+-- lib/
|   +-- ojdbc6.jar
+-- README.md

# Compilation: 

Dans la racine: javac -cp "./lib/ojdbc6.jar" -d bin src/*.java
# Exécution:

Dans la racine: java -cp "bin:./lib/ojdbc6.jar" Main
</pre>

